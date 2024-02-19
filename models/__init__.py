import logging
logger = logging.getLogger('base')


def create_model(opt):
    # image restoration
    model = opt['model']
    if model == 'sr':
        from .SIEN_model import SIEN_Model as M
    elif model == 'multi':
        from .multitask_SIEN_model import multitask_SIEN_model as M
    elif model == 'UNet' :
        from .multitask_UNet_model import multitask_UNet_model as M
    elif model == 'ViT' :
        from .multitask_ViT_model import multitask_ViT_model as M
    elif model == 'BCDU' :
        from .multitask_BCDU_model  import multitask_BCDUNet_model as M
    elif model == 'DEGAN' :
        from .multitask_DeGAN_model  import multitask_DEGAN_model as M
    # to visualize shallow feature map
    elif model == 'DocNC' :
        from .multitask_docnc_model  import multitask_docnc_model as M
    elif model == 'Barlow' :
        from .multitask_Barlow_model_new  import multitask_Barlow_model as M
    elif model == 'DIAE' :
        from .multitask_DIAE_model  import multitask_DIAE_model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

