import numpy as np
import onnx
import onnxruntime as rt
import cv2

input_image_path = 'im1501.jpg'
ONNX_Model_Path = 'resnet18_imagenet.onnx'


def onnx_infer(im, onnx_model):
    # InferenceSession获取onnxruntime解释器
    sess = rt.InferenceSession(onnx_model)
    # 模型的输入输出名，必须和onnx的输入输出名相同，可以通过netron查看，如何查看参考下文
    ort_inputs = {'input': im}

    output_name = ['boxes', 'labels', 'scores', 'masks']
    # run方法用于模型推理，run(输出张量名列表，输入值字典)
    pred_logits = sess.run(['boxes'], ort_inputs)[0]
    print(pred_logits)
    boxes = output[0]
    labels = output[1]
    scores = output[2]
    masks = output[3]


if __name__ == '__main__':
    # 图片的预处理
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)  # 3维转4维

    output = onnx_infer(img, ONNX_Model_Path)


# import torch
# from torchvision import models
#
# # 有 GPU 就用 GPU，没有就用 CPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)
# model = models.resnet18(pretrained=True)
# model = model.eval().to(device)
# print(model)
# x = torch.randn(1, 3, 256, 256).to(device)
# output = model(x)
# print(output.shape)
# with torch.no_grad():
#     torch.onnx.export(
#         model,                       # 要转换的模型
#         x,                           # 模型的任意一组输入
#         'resnet18_imagenet.onnx',    # 导出的 ONNX 文件名
#         opset_version=11,            # ONNX 算子集版本
#         input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
#         output_names=['output']      # 输出 Tensor 的名称（自己起名字）
#     )