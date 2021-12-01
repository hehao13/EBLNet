"""
Network Initializations
"""

import logging
import importlib
import torch


def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion, args=args)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion, args):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    if model == 'EBLNet_resnet50_os8' or model == 'EBLNet_resnet50_os16' or \
            model == 'EBLNet_resnet101_os8' or model == 'EBLNet_resnext101_os8':
        net = net_func(num_classes=num_classes, criterion=criterion,
                       num_cascade=args.num_cascade, num_points=args.num_points, threshold=args.thres_gcn)
    else:
        net = net_func(num_classes=num_classes, criterion=criterion)
    return net
