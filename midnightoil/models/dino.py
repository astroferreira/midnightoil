import tensorflow as tf

from ..dinoloss import DinoLoss
from .transformers import SwinTransformer

class Dino(tf.keras.models.Model):
    def __init__(
        self, teacher_model, student_model, student_weights=None, teacher_weights=None
    ):
        super(Dino, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.student_weights = student_weights
        self.teacher_weights = teacher_weights
        self.dino_loss = DinoLoss()

    def compile(self, optimizer):
        super(Dino, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        global_image, local_image = data        
        #local_image =  sum(local_image, ())
        #global_image = sum(global_image, ())
        #local_image = tf.stack(local_image)
        #global_image = tf.stack(global_image)
        global_image = tf.reshape(global_image, shape=(4, 128, 128, 1))
        local_image = tf.reshape(local_image, shape=(16, 64, 64, 1))
        with tf.GradientTape() as tape:
            teacher_output = self.teacher_model(global_image)
            student_output = self.student_model(local_image)
            loss = tf.reduce_mean(self.dino_loss(student_output, teacher_output))
            student_gradients = tape.gradient(
                loss, self.student_model.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(student_gradients, self.student_model.trainable_variables)
            )
            return {"loss": loss}

    def test_step(self, data):
        global_image, local_image = data
        global_image = tf.reshape(global_image, shape=(4, 128, 128, 1))
        local_image = tf.reshape(local_image, shape=(16, 64, 64, 1))
        #local_image = sum(local_image, ())
        #global_image = sum(global_image, ())
        #local_image = tf.stack(local_image)
        #global_image = tf.stack(global_image)
        #global_image = tf.reshape(global_image, shape=(global_image.shape[0], 128, 128, 1))
        #local_image = tf.reshape(local_image, shape=(local_image.shape[0], 64, 64, 1))
        teacher_output = self.teacher_model(global_image, training=False)
        student_output = self.student_model(local_image, training=False)

        loss = tf.reduce_mean(self.dino_loss(student_output, teacher_output))

        return {"loss": loss}

    def call(self, image):
        output = self.teacher_model(image, training=False)


import tensorflow_addons as tfa


class DinoHead(tf.keras.models.Model):
    def __init__(
        self,
        in_dim=192,
        out_dim=2048,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=128,
        bottleneck_dim=64,
    ):
        super(DinoHead, self).__init__()
        self.in_dim = in_dim
        self.use_bn = use_bn
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.norm_last_layer = norm_last_layer
        self.last_layer = tf.keras.layers.Dense(self.out_dim)

        self.mlp_block = self.mlp()

    def mlp(self):
        layer = []
        layer.append(tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.in_dim,)))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tfa.layers.GELU())
        for _ in range(self.nlayers - 2):
            layer.append(tf.keras.layers.Dense(self.hidden_dim))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tfa.layers.GELU())
        layer.append(tf.keras.layers.Dense(self.bottleneck_dim))
        return tf.keras.Sequential(layer)

    def call(self, input_tensor, training=None):
        x = self.mlp_block(input_tensor, training)
        x = tf.nn.l2_normalize(x, axis=-1)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(tf.keras.models.Model):
    def __init__(self, backbone, head, weights=None):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone
        if weights:
            try:
                print("Restoring model weights from: ", weights)
                self.load_weights(weights)
            except Exception:
                raise ValueError

    @staticmethod
    def unique_consecutive(x):
        neq = tf.math.not_equal(x, x)
        neq = tf.cast(neq, tf.int32)
        if neq.shape[0] > 1:
            neq = tf.math.cumsum(tf.cast(neq, tf.int32), axis=0)
        neq = tf.concat([[0], neq], axis=0)
        _, _, count = tf.unique_with_counts(neq)
        return count

    def call(self, x):
        if not isinstance(x, list):
            x = [x]
        unq = tf.constant([inp.shape[0] for inp in x], dtype=tf.int32)
        count = self.unique_consecutive(unq)
        start_idx, output = tf.constant(0), tf.zeros((0, 192), dtype=tf.float32)
        for end_idx in count:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(output, tf.TensorShape([None, None]))]
            )
            _out = self.backbone(
                x[0][tf.get_static_value(start_idx) : tf.get_static_value(end_idx)]
            )
            if isinstance(_out, tuple):
                _out = _out[0]
            output = tf.concat([output, _out], axis=0)
            start_idx = end_idx
        return self.head(output)


def load_base(cfg, image_size):

    cfg['input_size'] = (image_size, image_size)
    cfg['include_top'] = False
    print(cfg)
    model = SwinTransformer(cfg)
    #model = vit.vit_b16(
    #    image_size=image_size,
    #    pretrained=include_pretrained,
    #    pretrained_top=False,
    #    include_top=False,
    #)
    return model