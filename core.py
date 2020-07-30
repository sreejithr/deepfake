import cv2
import torch
import yaml
import imageio
import throttle
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from skimage.transform import resize
from scipy.spatial import ConvexHull
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback

#from animate import normalize_kp
# command = [ffmpeg,
#     '-y',
#     '-f', 'rawvideo',
#     '-vcodec','rawvideo',
#     '-pix_fmt', 'bgr24',
#     '-s', dimension,
#     '-i', '-',
#     '-c:v', 'libx264',
#     '-pix_fmt', 'yuv420p',
#     '-preset', 'ultrafast',
#     '-f', 'flv',
#     'rtmp://10.10.10.80/live/mystream']

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

@throttle.wrap(1, 2)
def forward(source_image, driving_frame, kp_source, kp_driving_initial, generator, kp_detector, relative=True, adapt_scale=True, cpu=True):
  kp_driving = kp_detector(driving_frame)
  kp_norm = normalize_kp(
    kp_source=kp_source,
    kp_driving=kp_driving,
    kp_driving_initial=kp_driving_initial,
    use_relative_movement=relative,
    use_relative_jacobian=relative,
    adapt_movement_scale=adapt_scale
  )
  out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
  return np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--config", required=True, help="path to config")
  parser.add_argument("--source_image", required=True, help="path to source image")
  parser.add_argument("--checkpoint", default="vox-cpk.pth.tar", help="path to checkpoint")
  parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
  parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
  parser.add_argument("--cpu", dest="cpu", action="store_true", help="CPU mode")
  parser.set_defaults(relative=False)
  parser.set_defaults(adapt_scale=False)
  opt = parser.parse_args()

  generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

  source_image = imageio.imread(opt.source_image)
  source_image = resize(source_image, (256, 256))[..., :3]

  source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
  if not opt.cpu:
    source = source.cuda()

  kp_source = kp_detector(source)

  #out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (256, 256))

  kp_driving_initial = None
  camera = cv2.VideoCapture(0)
  ret, frame = camera.read()

  while True:
    ret, frame = camera.read()
    resized = resize(frame, (256, 256))[..., :3]

    if not opt.cpu:
      resized = resized.cuda()

    # y = torch.tensor(np.array(resized))
    # x = y.cpu().numpy()
    # image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # # x = y.permute(1, 2, 0)
    # plt.imshow(np.array(image))
    # plt.show()

    driving_resized = torch.tensor(np.array(resized)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not kp_driving_initial:
      kp_driving_initial = kp_detector(driving_resized)

    fake_frame = forward(
      source,
      driving_resized,
      kp_source,
      kp_driving_initial,
      generator,
      kp_detector,
      relative=opt.relative,
      adapt_scale=opt.adapt_scale,
      cpu=opt.cpu   
    )
    cv2.imshow("frame", fake_frame)

    #x = np.squeeze(driving_resized, axis=(0,))
    #x = driving_resized[0].permute(1, 2, 0)
    # plt_driving = driving_resized #permute(2, 3, 1)
    #print(plt_driving.shape)
    #plt.imshow(x)
    #plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):   
      break

  camera.release()
  cv2.destroyAllWindows()
