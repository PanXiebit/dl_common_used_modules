import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        # conv1, return tensor with shape [batch, 256, 20, 20]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256,
                               kernel_size=9, stride=1)

        # primary capsules layer, return tensor with shape [batch, 1152, 8, 1]
        self.primaryCaps = CapsLayer(num_inputs=256,
                                     num_outputs=32,
                                     vec_len_in=1,
                                     vec_len_out=8,
                                     with_routing=False,
                                     layer_type="CONV")

        # DigitCaps layer, return shape [batch, 10, 16, 1]
        self.digitCaps = CapsLayer(num_inputs=6*6*32,
                                   num_outputs=10,
                                   vec_len_in=8,
                                   vec_len_out=16,
                                   with_routing=True,
                                   layer_type="FC")

    def forward(self, x):
        x = self.conv1(x)
        x = self.primaryCaps(x)
        out = self.digitCaps(x)
        return out


class CapsLayer(nn.Module):
    """ Capsule Layer
    Args:
        input: a 4-D tensor
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.
    """
    def __init__(self, num_inputs, num_outputs, vec_len_in, vec_len_out, with_routing, layer_type="FC"):
        super(CapsLayer, self).__init__()
        self.num_outputs = num_outputs
        self.vec_len_in = vec_len_in
        self.vec_len_out = vec_len_out
        self.with_routing = with_routing
        self.layer_type = layer_type

        if self.layer_type == "CONV" and not self.with_routing:
            kernel_size = 9
            stride = 2
            self.conv = nn.Conv2d(in_channels=num_inputs, out_channels=num_outputs*vec_len_out,
                                  kernel_size=kernel_size, stride=stride)

        if self.layer_type == "FC" and self.with_routing:
            # weight, W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
            # num_caps_1 = num_inputs = 1152
            self.W = nn.Parameter(torch.randn(1, num_inputs, num_outputs*vec_len_out, vec_len_in, 1))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_outputs, vec_len_out, 1))


    # def __call__(self, kernel_size, ):
    def forward(self, x):
        if self.layer_type == "CONV" and not self.with_routing:
            """
            x: [batch, 256, 20, 20]
            return: [batch, 1152, 8, 1]
            """
            capsules = self.conv(x)   # [batch, 32*8, 6, 6]
            capsules = capsules.view(capsules.size(0), -1, self.vec_len_out, 1)   # [batch, 1152, 8, 1]
            capsules = self.squash(capsules, dim=-2)


        elif self.layer_type == "FC" and self.with_routing:
            """
            x: [batch, 1152,8, 1]
            return: [batch, 10, 16, 1]
            """

            x = x.view(x.size(0), -1, 1, x.size(-2), 1)   # [batch, 1, 1152, 8, 1]
            # b_IJ mean the sum of every capsules.
            b_IJ = torch.zeros(x.size(0), x.size(1), self.num_outputs, 1, 1)  # [batch, 1152, 10, 1, 1]
            capsules = self.routing(x, b_IJ, num_outputs=self.num_outputs, num_dim=self.vec_len_out)
            capsules = capsules.squeeze()
        else:
            raise ImportError("the layer type must be CONV or FC")
        return capsules

    @staticmethod
    def squash(x, dim=-2):
        mag_squ = torch.sum(x**2, dim, keepdim=True)
        mag = torch.sqrt(mag_squ)
        return (mag_squ / (1.0 + mag_squ)) * (x/mag)


    def routing(self, input, b_IJ, num_outputs=10, num_dim=10):
        """ The routing algorithm

        :param input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        :param b_IJ: the initialization of weight for u_hat
        :param num_outputs: the number of output capsules.
        :param num_dim: the number of dimensions for output capsule.
        :return:
        """

        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        # [batch, 1152, 1, 8, 1] * [batch, 1152, 160, 1, 1]
        u_hat = torch.sum(self.W * input, dim=3, keepdim=True)   # [batch, 1152, 160, 1, 1]
        u_hat = u_hat.view(u_hat.size(0), input.size(1), num_outputs, num_dim, 1) # [batch, 1152, 10, 16, 1]

        routing_iters = 3
        for r_iter in range(routing_iters):
            # line 4  c_ij <- softmax(b_ij)
            # computer weights on the num_outputs dim. means every num_inputs capsules's
            # contribution to the output capsules.
            # => [batch, 1152, 10, 1, 1]
            c_IJ = torch.softmax(b_IJ, dim=2)  # [batch, 1152, 10, 1, 1]
            if r_iter != routing_iters-1:
                # line 5
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch, 1152, 10, 16, 1]
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = torch.sum(c_IJ * u_hat, dim=1, keepdim=True) + self.bias

                # line 6
                v_J = self.squash(s_J, dim=-2)

                # line 7
                v_J_tiled = v_J.repeat(1, input.size(1), 1, 1, 1)
                u_produce_v = torch.sum(u_hat * v_J_tiled, dim=3, keepdim=True)
                # print(u_produce_v.shape)
                # assert u_produce_v.size() == [input.size(0), 1152, 10, 1, 1]

                b_IJ += u_produce_v
            else:
                s_J = torch.sum(c_IJ * u_hat, dim=1, keepdim=True) + self.bias
                v_J = self.squash(s_J, dim=-2)
        return v_J

if __name__ == "__main__":
    # x = torch.randn(5, 256, 20, 20)
    # priCaps = CapsLayer(num_inputs=256, num_outputs=32, vec_len_in=1,
    #                     vec_len_out=8, with_routing=False, layer_type="CONV")
    # x = priCaps(x)
    # # print(x.shape)
    #
    # DigitCaps = CapsLayer(num_inputs=1152, num_outputs=10, vec_len_in=8,
    #                       vec_len_out=16, with_routing=True, layer_type="FC")
    # out = DigitCaps(x)
    # print(out.shape)


    #
    x = torch.randn(5, 1, 28, 28)
    model = CapsuleNet()
    out = model(x)
    print(out.shape)
