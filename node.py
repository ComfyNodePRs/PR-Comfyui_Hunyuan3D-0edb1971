import os
import shutil
import warnings
import sys
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)
from huggingface_hub import snapshot_download
import torch
import numpy as np
from einops import rearrange
from PIL import Image, ImageSequence, ImageOps, ImageColor
from io import BytesIO
import ipywidgets as widgets
from IPython.display import display
import tempfile
import torch.nn.functional as F
from typing import Tuple
import folder_paths

# 导入您的 infer 模块中的必要类和函数
from .infer import seed_everything, save_gif
from .infer import Removebg, Image2Views, Views2Mesh, GifRenderer
# 添加当前目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class Hunyuan3DNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0}),
                "step": ("INT", {"default": 50}),
                "max_number_of_faces": ("INT", {"default": 90000}),
                "do_texture_mapping": ("BOOLEAN", {"default": False}),
                "do_render_gif": ("BOOLEAN", {"default": False}),
                "use_lite": ("BOOLEAN", {"default": False}),  # 新增参数
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING", )
    RETURN_NAMES = ("rmbg_image", "multiview_image", "obj_file_path", "glb_file_path", "gif_file_path", )
    FUNCTION = "imgTo3D"
    CATEGORY = "Image/3D"

    def __init__(self):
        # 移除 super().__init__() 调用
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 将模型初始化为 None，待执行时再加载
        self.worker_xbg = None
        self.worker_i2v = None
        self.worker_v23 = None
        self.worker_gif = None

    @staticmethod
    def download_model_files(repo_id: str, local_dir: str):
        """
        从 Huggingface 仓库下载所有文件到指定的本地目录。

        :param repo_id: Huggingface 仓库的 ID，例如 "tencent/Hunyuan3D-1"
        :param local_dir: 本地目标目录，例如 "weights"
        """
        print(f"模型文件未找到，正在从 Huggingface 仓库 {repo_id} 下载所有文件到 {local_dir} ...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type='model', ignore_patterns=["*.gitignore"])
            print("模型文件下载完成。")
        except Exception as e:
            print(f"下载模型文件时出错: {e}")
            raise

    def imgTo3D(self, image, seed, step, max_number_of_faces, do_texture_mapping, do_render_gif, use_lite):
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                print(f"Original tensor shape: {image.shape}")
                if image.dim() == 4 and image.shape[0] == 1:
                    image = image.squeeze(0)
                    print(f"Squeezed tensor shape: {image.shape}")
                if image.dim() == 3:
                    if image.shape[2] in [1, 3, 4]:
                        print(f"Image is in [H, W, C] format with shape: {image.shape}")
                        pass
                    elif image.shape[0] in [1, 3, 4]:
                        print(f"Image is in [C, H, W] format with shape: {image.shape}")
                        image = image.permute(1, 2, 0)
                        print(f"Transposed image shape: {image.shape}")
                    else:
                        raise ValueError(f"Unsupported image shape: {image.shape}")
                else:
                    raise ValueError(f"Unsupported image dimensions: {image.dim()}")
                image = image.cpu().numpy()
                print(f"Converted to numpy array, shape: {image.shape}, dtype: {image.dtype}")
                if image.dtype in [np.float32, np.float64]:
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                    print(f"Scaled image to uint8, dtype: {image.dtype}")
                elif image.dtype == np.uint8:
                    print(f"Image is already uint8, dtype: {image.dtype}")
                else:
                    image = image.astype(np.uint8)
                    print(f"Converted image to uint8, dtype: {image.dtype}")
                if image.ndim != 3 or image.shape[2] not in [1, 3, 4]:
                    raise ValueError(f"Invalid image shape after processing: {image.shape}")
                image = Image.fromarray(image)
                print(f"Converted to PIL Image, size: {image.size}, mode: {image.mode}")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        else:
            print(f"Input image is already a PIL Image, size: {image.size}, mode: {image.mode}")  

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "weights")
        svrm_dir = os.path.join(weights_dir, "svrm")
        weights_path = os.path.join(svrm_dir, "svrm.safetensors")

        # 检查模型权重文件是否存在
        if not os.path.exists(weights_path):
            print(f"未找到模型文件 {weights_path}，开始下载模型文件...")
            repo_id = "tencent/Hunyuan3D-1"  # Huggingface 仓库 ID
            self.download_model_files(repo_id=repo_id, local_dir=weights_dir)
            # 检查下载后是否存在模型文件
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"下载后仍未找到模型文件 {weights_path}。请检查仓库 ID 或网络连接。")

        if self.worker_xbg is None:
            self.worker_xbg = Removebg()
        if self.worker_i2v is None:
            self.worker_i2v = Image2Views(
                use_lite=use_lite,  
                device=self.device
            )
        if use_lite:
            config_path = os.path.join(base_dir, "svrm/configs/svrm.yaml")
            weights_path = os.path.join(base_dir, "weights/svrm/svrm.safetensors")
        else:
            config_path = os.path.join(base_dir, "svrm/configs/svrm.yaml")
            weights_path = os.path.join(base_dir, "weights/svrm/svrm.safetensors")

        if self.worker_v23 is None:
            self.worker_v23 = Views2Mesh(
                config_path,
                weights_path,
                use_lite=use_lite,  
                device=self.device
            )
        if self.worker_gif is None:
            self.worker_gif = GifRenderer(self.device)

        save_folder = self.prepare_save_folder()

        try:
            # 阶段 1：去除背景
            rgba = self.worker_xbg(image)
            rgba.save(os.path.join(save_folder, 'img_nobg.png'))

            # 阶段 2：生成多视图图像
            res_img, pils = self.worker_i2v(rgba, seed, step)
            save_gif(pils, os.path.join(save_folder, 'views.gif'))
            views_img, cond_img = res_img[0], res_img[1]

            # 准备显示的多视图图像
            img_array = np.asarray(views_img, dtype=np.uint8)
            show_img = rearrange(img_array, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            show_img = show_img[self.worker_i2v.order, ...]
            show_img = rearrange(show_img, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
            show_img = Image.fromarray(show_img)

            # 阶段 3：从视图生成网格
            self.worker_v23(
                views_img, cond_img,
                seed=seed, save_folder=save_folder,
                target_face_count=max_number_of_faces,
                do_texture_mapping=do_texture_mapping
            )

            # 阶段 4：渲染 GIF
            gif_file_path = None
            if do_render_gif:
                self.worker_gif(
                    os.path.join(save_folder, 'mesh.obj'), 
                    gif_dst_path=os.path.join(save_folder, 'output.gif')
                )
                gif_file_path = os.path.join(save_folder, 'output.gif')
                

            # 准备输出
            rmbg_image = np.array(rgba, dtype=np.uint8)  # [H, W, C]
            multiview_image = np.array(show_img, dtype=np.uint8)  # [H, W, C]
            obj_file_path = os.path.join(save_folder, 'mesh_with_colors.obj')
            glb_file_path = os.path.join(save_folder, 'mesh.glb')
            
            # Convert to torch.Tensor, keep dimensions [H, W, C], normalize to [0, 1]
            rmbg_image_tensor = torch.from_numpy(rmbg_image).float() / 255.0  # [H, W, C]
            multiview_image_tensor = torch.from_numpy(multiview_image).float() / 255.0  # [H, W, C]

            # Add batch dimension
            rmbg_image_tensor = rmbg_image_tensor.unsqueeze(0)  # [1, H, W, C]
            multiview_image_tensor = multiview_image_tensor.unsqueeze(0)  # [1, H, W, C]

            # Return
            outputs = (
                rmbg_image_tensor,
                multiview_image_tensor,
                obj_file_path,
                glb_file_path,
                gif_file_path,
            )
            return outputs

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise

    def prepare_save_folder(self):
        # 获取当前文件的上级再上一级目录
        base_dir = folder_paths.get_output_directory()
        
        # 组合新的保存路径
        output_dir = os.path.join(base_dir, '3D_output')
        os.makedirs(output_dir, exist_ok=True)

        # 检查已有的子文件夹
        exists = set(int(_) for _ in os.listdir(output_dir) if not _.startswith(".") and _.isdigit())
        if len(exists) == 30:
            shutil.rmtree(os.path.join(output_dir, "0"))
            cur_id = 0
        else:
            cur_id = min(set(range(30)) - exists)

        # 删除循环中下一个要用的文件夹
        next_folder = os.path.join(output_dir, str((cur_id + 1) % 30))
        if os.path.exists(next_folder):
            shutil.rmtree(next_folder)

        # 创建当前ID的保存文件夹
        save_folder = os.path.join(output_dir, str(cur_id))
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))        
        
class SquareImage:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 1024, "min": 8, "max": 8096, "step": 16}),
                "upscale_method": (cls.upscale_methods,),
                "padding_color": ("COLOR", {"default": (255, 255, 255)}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Square_Image",)
    FUNCTION = "make_square"
    CATEGORY = "TTP/Image"

    def make_square(self, image, resolution, upscale_method, padding_color=(255, 255, 255)):
        ret_images = []

        # 映射用户选择的缩放方法到 PIL 的 resample 方法
        resample_methods = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.BOX,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }

        resample = resample_methods.get(upscale_method, Image.NEAREST)

        for img_tensor in image:
            # 将张量转换为 PIL 图像
            pil_image = tensor2pil(img_tensor)

            # 获取原始尺寸
            original_width, original_height = pil_image.size

            # 判断是否已经是正方形
            if original_width == original_height:
                square_image = pil_image
            else:
                # 计算填充尺寸
                max_side = max(original_width, original_height)
                delta_w = max_side - original_width
                delta_h = max_side - original_height
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

                # 填充图像
                square_image = ImageOps.expand(pil_image, padding, fill=padding_color)

            # 缩放到指定分辨率
            square_image = square_image.resize((resolution, resolution), resample=resample)

            # 转换回张量
            tensor_img = pil2tensor(square_image)
            ret_images.append(tensor_img)

        # 将处理后的图像张量堆叠起来
        return (torch.cat(ret_images, dim=0),)

class GifImageViewerNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gif_file_path": ("STRING", {"default": '', "multiline": False, "forceInput": True}),
            },
        }

    # 明确指定返回类型，这里假设返回的是 UI_IMAGE
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_gif_image"
    CATEGORY = "Image/Visualize"

    def preview_gif_image(self, gif_file_path):
    
        filename, subdirectory = self.get_fn_subf(gif_file_path)
        # 准备返回结果
        results = [
            {
                "filename": filename,  # 使用临时文件的路径
                "subfolder": subdirectory,  # 如果有子文件夹，填写子文件夹名称，否则为空字符串
                "type": "output"
            }
        ]

        return { "ui": { "images": results } }
        
    def get_fn_subf(self, gif_file_path):
        output_directory = folder_paths.get_output_directory()
        filename = os.path.basename(gif_file_path)
        directory_path = os.path.dirname(gif_file_path)
        subdirectory = os.path.relpath(directory_path, output_directory)

        return filename, subdirectory
         
# Register the node
NODE_CLASS_MAPPINGS = {
    "Hunyuan3DNode": Hunyuan3DNode,
    "SquareImage" : SquareImage,
    "GifImageViewerNode" : GifImageViewerNode, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DNode": "TTP_Hunyuan3DNode",
    "SquareImage": "TTP_SquareImage", 
    "GifImageViewerNode": "TTP_GIFViewer", 
}
