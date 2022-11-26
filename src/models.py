import torchvision.models as models
import segmentation_models_pytorch as smp

# Credit for core hrnet: https://github.com/HRNet/HRNet-Semantic-Segmentation
from .high_resolution_net.lib.models.hrnet import hrnet48, hrnet32, hrnet18


def make_unet(backbone='resnet34', weigths='imagenet',
              num_classes=1, activation='sigmoid', channels=3):

    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=weigths,
        classes=num_classes,
        activation=activation,
        in_channels=channels)

    return model


def make_unet_plusplus(backbone='resnet50', weigths='imagenet',
                   num_classes=1,
                   activation='sigmoid',channels=3):

    model = smp.UnetPlusPlus(
        encoder_name=backbone,
        encoder_weights=weigths,
        classes=num_classes,
        activation=activation,
        in_channels=channels)

    return model


def make_deeplabv3(backbone='resnet50', weigths='imagenet',
                   num_classes=1, activation='sigmoid', *args, **kwargs):

    model = smp.DeepLabV3(
        encoder_name=backbone,
        encoder_weights=weigths,
        classes=num_classes,
        activation=activation)

    return model


def make_deeplab(out_channels=1, pretrained=True):

    if (pretrained is not None) and (pretrained is not False):
        try:
            model = models.segmentation.deeplabv3_resnet50(
                pretrained=True)
        except Exception as e:
            print(e)
            model = models.segmentation.deeplabv3_resnet50(
                pretrained=False)
    else:
        model = models.segmentation.deeplabv3_resnet50(
            pretrained=False)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(
        2048, num_classes=out_channels)

    model.train()
    return model


def make_hrnet_48(channels=3):
    model = hrnet48(pretrained=False, progress=True, channels=channels)
    return model


def make_hrnet_32(channels=3, **kwargs):
    model = hrnet32(pretrained=False, progress=True, channels=channels, **kwargs)
    return model


def make_hrnet_18(channels=3):
    model = hrnet18(pretrained=False, progress=True, channels=channels)
    return model
