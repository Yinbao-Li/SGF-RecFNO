import configargparse

from utils.iso_geometry import DEFAULT_QUANTILES


def parse_iso_args():
    parser = configargparse.ArgumentParser(description='IsoRecFNO heat field reconstruction')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='iso_recfno_heat_25')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt')
    parser.add_argument('--tb_path', type=str, default='logs/tb')
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--sensor_num', type=int, default=25)
    parser.add_argument('--fc_h', type=int, default=12)
    parser.add_argument('--fc_w', type=int, default=12)
    parser.add_argument('--out_h', type=int, default=200)
    parser.add_argument('--out_w', type=int, default=200)
    parser.add_argument('--modes1', type=int, default=50)
    parser.add_argument('--modes2', type=int, default=50)
    parser.add_argument('--width', type=int, default=32)

    parser.add_argument('--train_end', type=int, default=4000)
    parser.add_argument('--val_end', type=int, default=5000)
    parser.add_argument('--quantiles', type=float, nargs='+', default=list(DEFAULT_QUANTILES))

    parser.add_argument('--field_loss', type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument('--lambda_grad', type=float, default=0.1)
    parser.add_argument('--lambda_sdf', type=float, default=0.5)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--sdf_scale', type=float, default=5.0)

    return parser.parse_args()


def parse_sgf_args():
    parser = configargparse.ArgumentParser(description='Self-Geometry Feedback RecFNO')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='sgf_recfno_heat_25')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt')
    parser.add_argument('--tb_path', type=str, default='logs/tb')
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--sensor_num', type=int, default=25)
    parser.add_argument('--fc_h', type=int, default=12)
    parser.add_argument('--fc_w', type=int, default=12)
    parser.add_argument('--out_h', type=int, default=200)
    parser.add_argument('--out_w', type=int, default=200)
    parser.add_argument('--modes1', type=int, default=50)
    parser.add_argument('--modes2', type=int, default=50)
    parser.add_argument('--width', type=int, default=32)

    parser.add_argument('--refine_modes1', type=int, default=24)
    parser.add_argument('--refine_modes2', type=int, default=24)
    parser.add_argument('--refine_width', type=int, default=32)

    parser.add_argument('--train_end', type=int, default=4000)
    parser.add_argument('--val_end', type=int, default=5000)
    parser.add_argument('--quantiles', type=float, nargs='+', default=list(DEFAULT_QUANTILES))

    parser.add_argument('--field_loss', type=str, default='l1', choices=['l1', 'mse', 'relative_l2'])
    parser.add_argument('--lambda_field', type=float, default=1.0)
    parser.add_argument('--lambda_grad', type=float, default=0.1)
    parser.add_argument('--lambda_sdf', type=float, default=0.5)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--sdf_scale', type=float, default=5.0)

    return parser.parse_args()


def parse_geo_enc_args():
    parser = configargparse.ArgumentParser(description='GeoEnc RecFNO heat field reconstruction')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='geo_enc_fno_heat_25')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt')
    parser.add_argument('--tb_path', type=str, default='logs/tb')
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--sensor_num', type=int, default=25)
    parser.add_argument('--fc_h', type=int, default=12)
    parser.add_argument('--fc_w', type=int, default=12)
    parser.add_argument('--out_h', type=int, default=200)
    parser.add_argument('--out_w', type=int, default=200)
    parser.add_argument('--modes1', type=int, default=50)
    parser.add_argument('--modes2', type=int, default=50)
    parser.add_argument('--width', type=int, default=32)

    parser.add_argument('--train_end', type=int, default=4000)
    parser.add_argument('--val_end', type=int, default=5000)
    parser.add_argument('--quantiles', type=float, nargs='+', default=list(DEFAULT_QUANTILES))

    parser.add_argument('--field_loss', type=str, default='l1', choices=['l1', 'mse', 'relative_l2'])
    parser.add_argument('--lambda_grad', type=float, default=0.1)
    parser.add_argument('--lambda_sdf', type=float, default=0.5)
    parser.add_argument('--sdf_scale', type=float, default=5.0)

    return parser.parse_args()


def parses():
    """Backward-compatible alias used by original RecFNO scripts."""
    parser = configargparse.ArgumentParser(description='field_reconstruction')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='recon')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt')
    parser.add_argument('--tb_path', type=str, default='logs/tb')
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()
