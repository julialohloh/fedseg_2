from base64 import encode
from math import floor
import torch
import torch.nn as nn


class AE(torch.nn.Module):
    """
    Fully Convolutional Network based AutoEncoder.

    """

    def calc_size(
        self, input_height,input_width, padding, dilation, kernel_size, stride, return_diff=False
    ):
        # assumes a square image, therefore height = width
        output_height = floor(
            (input_height + (2 * padding) - dilation * (kernel_size[0] - 1) - 1) / stride
            + 1
        )

        output_width = floor(
            (input_width + (2 * padding) - dilation * (kernel_size[1] - 1) - 1) / stride
            + 1
        )

        height_diff = input_height - output_height
        width_diff = input_width - output_width

        print(f"For height, with padding: {padding}, dilation: {dilation}, kernel_size[0]: {kernel_size[0]}, and stride:{stride}, the output height is {output_height}, and the difference is {height_diff}")
        print(f"For width, with padding: {padding}, dilation: {dilation}, kernel_size[0]: {kernel_size[0]}, and stride:{stride}, the output width is {output_width}, and the the difference is {width_diff}")
        if return_diff:
            return output_height,output_width, height_diff, width_diff
        else:
            return output_height,output_width

    def calc_size_decoder(
        self,
        input_height,
        input_width,
        padding,
        output_padding,
        dilation,
        kernel_size,
        stride,
        return_diff=False,
    ):
        # assumes a square image, therefore height = width
        output_height = (
            (input_height - 1) * stride
            - (2 * padding)
            + dilation * (kernel_size[0] - 1)
            + output_padding[0]
            + 1
        )

        output_width = (
            (input_width - 1) * stride
            - (2 * padding)
            + dilation * (kernel_size[1] - 1)
            + output_padding[1]
            + 1
        )
        height_diff = output_height - input_height
        width_diff = output_width - input_width
        if return_diff:
            return output_height,output_width, height_diff, width_diff
        else:
            return output_height,output_width


    def generate_encoder_layers(
        self,
        input_height,
        input_width,
        encoded_size, # What size to encode the image to
        in_channels, # Num of channels in the input
        out_channels, # Num of channels in the output to the user defined model => How many channels does the user defined model
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        dilation=1,
        groups=1,
        bias=False,
    ):
        """
        Does the following in order:-
            1. Check if the input size is at least 2x that of the desired encoded size
                - If it is, we will add Conv2D layers, and Maxpooling layers to halve the spatial dimensions
                - If the input size is NOT at least 2x that of the desired encoded size, we will proceed without Maxpooling layers
            2. Check if the cur/input size is greater than 2
                - If it is, we will add #n Conv2D layers that will reduce the spatial dimensions of the input by 2 where
                #n is the greatest multiple of 2
                - If it is not, we will proceed without adding any -2 Conv2D layers
            3. Check if the cur/input size is greater than 1
                - If it is, we will add #n Conv2D layers that will reduce the spatial dimensions of the input by 1 where
                #n is the number of times it takes to reduce the input to the desired encoded size
        """
        
        layers = nn.Sequential()
        # Assumes a square image
        reduction = None
        round = 0
        count = 1
        cur_image_height = input_height
        cur_image_width = input_width
        assert input_height == input_width, "Input height must equal input width."

        # Asset that input weight/width are multiples of each other
        # if input_height >= input_width:
        #     assert input_height % input_width == 0 ,"Input height should equal input_width or be an integer multiple"
        # elif input_width >= input_height:
        #     assert input_width % input_height == 0 , "Input width should equal input_height or be an integer multiple"

        print(f"Starting the encoder layers... Original input height is {input_height} and input width is {input_width}")
        # Check if the input image is at least 2x the desired encoded size
        next_iteration_img_size = input_height // 2
        # Adds Conv2D and Maxpool layers while the image is at least 2x greater than the desired encoded size
        while next_iteration_img_size > encoded_size:
            # If first round, in_channel = in_channel
            if round == 0:
                count = 1
                layers.add_module(
                    f"Conv{round}-{count}",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                )
                cur_image_height,cur_image_width,height_reduction,width_reduction = self.calc_size(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    stride=stride,
                    return_diff=True
                )

                print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                
                layers.add_module(
                    f"BatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                )
                layers.add_module(f"Relu{round}-{count}", nn.ReLU(False)),

                count += 1

                if cur_image_height - height_reduction >= encoded_size:
                    layers.add_module(
                        f"Conv{round}-{count}",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                    )

                    cur_image_height,cur_image_width = self.calc_size(
                        input_height=cur_image_height,
                        input_width=cur_image_width,
                        padding=padding,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                    print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")

                    layers.add_module(
                        f"BatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                    )
                    layers.add_module(f"Relu{round}-{count}", nn.ReLU(False)),

                if cur_image_height // 2 >= encoded_size:
                    layers.add_module(f"MaxPool{round}", nn.MaxPool2d(kernel_size=2))
                    cur_image_height = cur_image_height //2
                    cur_image_width = cur_image_width // 2
                    print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")
            else:
                count = 1
                if cur_image_height - height_reduction >= encoded_size:
                    layers.add_module(
                        f"Conv{round}-{count}",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                    )

                    cur_image_height,cur_image_width = self.calc_size(
                        input_height=cur_image_height,
                        input_width=cur_image_width,
                        padding=padding,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        stride=stride,
                    )

                    print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")

                    layers.add_module(
                        f"BatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                    )
                    layers.add_module(f"Relu{round}-{count}", nn.ReLU(False)),

                    count += 1
                if cur_image_height - height_reduction >= encoded_size:
                    layers.add_module(
                        f"Conv{round}-{count}",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                    )
                    
                    cur_image_height,cur_image_width = self.calc_size(
                        input_height=cur_image_height,
                        input_width=cur_image_width,
                        padding=padding,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                    print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")

                    layers.add_module(
                        f"BatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                    )
                    layers.add_module(f"Relu{round}-{count}", nn.ReLU(False)),
                if cur_image_height // 2 >= encoded_size:
                    layers.add_module(f"MaxPool{round}", nn.MaxPool2d(kernel_size=2))
                    cur_image_height = cur_image_height //2
                    cur_image_width = cur_image_width // 2
                    print(f"Round {round} - Count {count}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")

            round += 1
            count += 1
            next_iteration_img_size = cur_image_height // 2

        print(f"Finished adding max pool layers\n")
        # Difference between image size and desired encoded size that we need to shrink after adding maxpooling layers above
        amt_height_to_shrink = int(cur_image_height - encoded_size)
        amt_width_to_shrink = int(cur_image_width - encoded_size)

        # we assume that the image is a square image where height == width
        _,_,height_reduction,width_reduction = self.calc_size(
            input_height=cur_image_height,
            input_width=cur_image_width,
            padding=padding,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
            return_diff=True,
        )

        num_convs = int(amt_height_to_shrink // height_reduction)
        print(f"num convs: {num_convs}\n")
        if num_convs > 0:
            print(f"Continuing encoder layers after maxpooling layers...")
            for i in range(num_convs):
                layers.add_module(
                    f"ConvFinal{i}-{count}",
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                )

                cur_image_height,cur_image_width = self.calc_size(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    stride=stride,
                )

                print(f"Round {i}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")

                layers.add_module(
                    f"BatchNormFinal{i}-{count}", nn.BatchNorm2d(out_channels)
                )
                layers.add_module(f"ReluFinal{i}-{count}", nn.ReLU(False))

        # Force down to desired shape if the above does not bring it down to desired shape already
        amt_height_to_force_shrink = int(cur_image_height - encoded_size)
        amt_width_to_force_shrink = int(cur_image_width - encoded_size)
        print(f"Amt height to force shrink: {amt_height_to_force_shrink}\n")
        if amt_height_to_force_shrink > 0:
            print(f"Continuing encoder layers... -1 image shrinking layers...")
            for i in range(amt_height_to_force_shrink):
                # Reduces size by 1
                layers.add_module(
                    f"ConvFinalShrink{i}-{count}",
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(2,2),
                        stride=1,
                        padding=0,
                        padding_mode="zeros",
                        dilation=1,
                        groups=groups,
                        bias=False,
                    ),
                )

                cur_image_height,cur_image_width = self.calc_size(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    dilation=dilation,
                    kernel_size=(2,2),
                    stride=stride,
                )

                print(f"Round {i}: cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                layers.add_module(
                    f"BatchNormFinalShrink{i}-{count}", nn.BatchNorm2d(out_channels)
                )
                layers.add_module(f"ReluFinalShrink{i}-{count}", nn.ReLU(False))

        return layers

    def generate_decoder_layers(
        # self,
        # input_height,
        # input_width,
        # decoded_size,
        # in_channels,
        # out_channels,
        # num_output_features,
        # kernel_size,
        # upsample_kernel_size=2,
        # stride=1,
        # upsample_stride=2,
        # padding=0,
        # output_padding=0,
        # padding_mode="zeros",
        # dilation=1,
        # groups=1,
        # bias=False,
        self,
        input_height,
        input_width,
        decoded_size,
        in_channels,
        out_channels,
        num_output_features,
        kernel_size,
        upsample_kernel_size,
        stride,
        upsample_stride,
        padding,
        output_padding,
        padding_mode,
        dilation,
        groups,
        bias,
    ):
        """
        Does the following:-
            1. Check if the input size is at least half that of the desired encoded size
                - If it is, we will add ConvTranspose2D layers, to double the spatial dimensions
                - If the input size is NOT at least half that of the desired encoded size, we will proceed without ConvTranspose2D layers
            2. Check if the cur/input & desired decoded size difference is greater than 2
                - If it is, we will add #n ConvTranspose2D layers that will increase the spatial dimensions of the input by 2 where
                #n is the greatest multiple of 2
                - If it is not, we will proceed without adding any +2 ConvTranspose2D layers
            3. Check if the cur/input & desired decoded size difference is greater than 1
                - If it is, we will add #n ConvTranspose2D layers that will increase the spatial dimensions of the input by 1 where
                #n is the number of times it takes to increase the input to the desired encoded size
            4. Check if the number of channels is the same as the desired output channels
                - If it is, nothing more is to be done
                - If it is NOt, we will do 1x1 convolutions to increase the number of channels to the # of desired output channels
        """
        layers = nn.Sequential()
        print(f"\nStarting the decoder layers... input height to decoder layers is {input_height} and input width is {input_width}")
        cur_image_height = input_height
        cur_image_width = input_width
        round = 0
        count = 1
        next_iteration_img_height = cur_image_height * 2
        next_iteration_img_width = cur_image_width * 2
        # We will double the spatial dimensions with COnvTranspose 2D
        while next_iteration_img_height <= decoded_size:
            print(f"Starting decoder ConvTranspose2D layers...Image height: {input_height}, Input width;{input_width}")
            if round == 0:
                count = 1
                # ConvTranspose2D reverses Maxpool func
                layers.add_module(
                    f"ConvTranspose2D{round}-{count}",
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                )
                cur_image_height,cur_image_width,height_increase,width_increase = self.calc_size_decoder(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    kernel_size=upsample_kernel_size,
                    stride=upsample_stride,
                    return_diff=True,
                )
                # print(f"Size after convtranspose:{cur_image_size},{increase}")
                print(f"Round {round} - Count {count}: Cur image sizes after ConvTranspose2D - cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                
                _,_,height_reduction,width_reduction = self.calc_size(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    stride=stride,
                    return_diff=True
                )
                if (cur_image_height - height_reduction) * 2 != cur_image_height:
                    layers.add_module(
                        f"UpsampleConv{round}-{count}",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                    )
                    cur_image_height,cur_image_width = self.calc_size(
                        input_height=cur_image_height,
                        input_width=cur_image_width,
                        padding=padding,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                    print(f"Round {round} - Count {count}: Cur image sizes after Conv2D downsampling- cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                    # print(f"Size after conv2d:{cur_image_size} ")
                    layers.add_module(
                        f"UpsampleBatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                    )
                    layers.add_module(f"UpsampleRelu{round}-{count}", nn.ReLU(False))

                # count += 1

                # layers.add_module(
                #     f"UpsampleConv{round}-{count}",
                #     nn.Conv2d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel_size=kernel_size,
                #         stride=stride,
                #         padding=padding,
                #         padding_mode=padding_mode,
                #         dilation=dilation,
                #         groups=groups,
                #         bias=bias,
                #     ),
                # )
                # cur_image_height,cur_image_width = self.calc_size(
                #     input_height=cur_image_height,
                #     input_width=cur_image_width,
                #     padding=padding,
                #     dilation=dilation,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # )
                # layers.add_module(
                #     f"UpsampleBatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                # )
                # layers.add_module(f"UpsampleRelu{round}-{count}", nn.ReLU(False))

            else:
                count = 1
                layers.add_module(
                    f"ConvTranspose2D{round}-{count}",
                    nn.ConvTranspose2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                )
                cur_image_height,cur_image_width,height_increase,width_increase = self.calc_size_decoder(
                    input_height=cur_image_height,
                    input_width=cur_image_width,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    kernel_size=upsample_kernel_size,
                    stride=upsample_stride,
                    return_diff=True,
                )
                print(f"Round {round} - Count {count}: Cur image sizes after ConvTranspose2D - cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                if (cur_image_height - height_reduction) * 2 != cur_image_height:
                    layers.add_module(
                        f"UpsampleConv{round}-{count}",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                        ),
                    )
                    cur_image_height,cur_image_width = self.calc_size(
                        input_height=cur_image_height,
                        input_width=cur_image_width,
                        padding=padding,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                    # print(f"Size after conv2d:{cur_image_size} ")
                    print(f"Round {round} - Count {count}: Cur image sizes after Conv2D downsampling - cur image height: {cur_image_height}, cur image width: {cur_image_width}")
                    layers.add_module(
                        f"UpsampleBatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                    )
                    layers.add_module(f"UpsampleRelu{round}-{count}", nn.ReLU(False))

                # count += 1

                # layers.add_module(
                #     f"UpsampleConv{round}-{count}",
                #     nn.Conv2d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel_size=kernel_size,
                #         stride=stride,
                #         padding=padding,
                #         padding_mode=padding_mode,
                #         dilation=dilation,
                #         groups=groups,
                #         bias=bias,
                #     ),
                # )
                # cur_image_height,cur_image_width = self.calc_size(
                #     input_height=cur_image_height,
                #     input_width=cur_image_width,
                #     padding=padding,
                #     dilation=dilation,
                #     kernel_size=kernel_size,
                #     stride=stride,
                # )
                # print(f"Size after conv2d:{cur_image_size} ")
                # layers.add_module(
                #     f"UpsampleBatchNorm{round}-{count}", nn.BatchNorm2d(out_channels)
                # )
                # layers.add_module(f"UpsampleRelu{round}-{count}", nn.ReLU(False))

            round += 1
            count = 1
            next_iteration_img_height = cur_image_height * 2

        print(f"Finished ConvTranspose2D layers...\n")
        # leftover to upsample after ConvTranspose2D layers taht double the size
        amt_height_to_upsample = decoded_size - cur_image_height
        amt_width_to_upsample = decoded_size - cur_image_width
        # We see how many +2 ConvTranspose2D layers we need to add
        num_conv_transpose = int(amt_height_to_upsample // 2)
        count = 1
        for i in range(num_conv_transpose):
            print(f"Continuing decoder layers after ConvTranspose2D layers...")
            layers.add_module(
                f"UpsampleConvTranspose2DFinal{i}-{count}",
                nn.ConvTranspose2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                ),
            )
            cur_image_height,cur_image_width = self.calc_size_decoder(
                input_height=cur_image_height,
                input_width = cur_image_width,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                kernel_size=(3,3),
                stride=1,
                return_diff=False,
            )
            print(f"Image sizes after +2 conv2d layers - image height:{cur_image_height}, image_width:{cur_image_width}")
            layers.add_module(
                f"UpsampleBatchNormFinal{i}-{count}", nn.BatchNorm2d(out_channels)
            )
            layers.add_module(f"UpsampleReluFinal{i}-{count}", nn.ReLU(False))

        # Force down to desired shape if the above does not bring it down to desired shape already
        # We see how many +1 ConvTranspose2D layers we need to add
        amt_to_force_upsample = int(decoded_size - cur_image_height)
        if amt_to_force_upsample > 0:
            print(f"Continuing decoder layers by adding + 1 layers")
            count = 1
            for i in range(amt_to_force_upsample):
                # Increase size by 1
                layers.add_module(
                    f"UpsampleConvTranspose2DFinalForced{i}-{count}",
                    nn.ConvTranspose2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(2,2),
                        stride=1,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    ),
                )
                cur_image_height,cur_image_width = self.calc_size_decoder(
                    input_height=cur_image_height,
                    input_width = cur_image_width,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    kernel_size=(2,2),
                    stride=1,
                    return_diff=False,
                )
                print(f"Size after +1 conv2d:{cur_image_height} ")
                layers.add_module(
                    f"UpsampleBatchNormFinalForced{i}-{count}",
                    nn.BatchNorm2d(out_channels),
                )
                layers.add_module(f"UpsampleReluFinalForced{i}-{count}", nn.ReLU(False))
        # We have reached the correct size via spatial upsampling, now we are going to convolve to get desired number of output channels/feature maps
        layers.add_module(
            f"1x1Conv2D",
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=num_output_features,
                kernel_size=(1,1),
                stride=1,
                padding=(0,0),
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )

        return layers

    # Default init values except the upsample variables(i.e upsample_kernel/stride) mirror the pytorch nn.Conv2d default param values
    def __init__(
        self,
        input_height,
        input_width,
        encoded_size,
        decoded_size,
        in_channels,
        out_channels,
        num_output_features,
        kernel_size,
        upsample_kernel_size=(2,2),
        stride=1,
        upsample_stride=2,
        padding=0,
        padding_mode="zeros",
        output_padding=(0,0),
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(AE, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.encoded_size = encoded_size
        self.decoded_size = decoded_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size
        self.upsample_kernel_size = upsample_kernel_size
        self.stride = stride
        self.upsample_stride = upsample_stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.encoder = self.generate_encoder_layers(
            input_height=self.input_height,
            input_width = self.input_width,
            encoded_size=self.encoded_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        self.decoder = self.generate_decoder_layers(
            input_height=self.encoded_size,
            input_width = self.encoded_size,
            decoded_size=self.decoded_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_output_features=self.num_output_features,
            kernel_size=self.kernel_size,
            upsample_kernel_size=self.upsample_kernel_size,
            stride=self.stride,
            upsample_stride=self.upsample_stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

    def forward(self, x):
        encoded = self.encoder(x)
#         print(f"shape after encoder layer pass is {encoded.shape}")
        if self.decoded_size <= self.encoded_size:
#             print(f"type after encoder layer pass is {encoded.type()}")
            return encoded
        else:
        # return encoded
            decoded = self.decoder(encoded)
#             print(f"type after decoder layer pass is {decoded.type()}")
            return decoded
