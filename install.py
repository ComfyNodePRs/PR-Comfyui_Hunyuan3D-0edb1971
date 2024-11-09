import sys
import os
import subprocess
import traceback

def add_current_dir_to_sys_path():
    """添加当前目录到 sys.path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        print(f"Added current directory to sys.path: {current_dir}")

def install_packages(python_executable):
    """安装所需的Python包"""
    try:
        # 定义要安装的包列表
        packages = [
            "diffusers",
            "transformers",
            "rembg",
            "tqdm",
            "omegaconf",
            "matplotlib",
            "opencv-python",
            "imageio",
            "jaxtyping",
            "einops",
            "SentencePiece",
            "accelerate",
            "trimesh",
            "PyMCubes",
            "xatlas",
            "libigl",
            "open3d"
        ]

        # 定义需要从GitHub安装的包
        git_packages = [
            "git+https://github.com/facebookresearch/pytorch3d",
            "git+https://github.com/NVlabs/nvdiffrast"
        ]

        # 安装普通包
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([
                python_executable, "-m", "pip", "install", package, 
            ])

        # 安装GitHub上的包
        for git_package in git_packages:
            print(f"Installing {git_package}...")
            subprocess.check_call([
                python_executable, "-m", "pip", "install", git_package
            ])

        print("所有依赖项已成功安装。")

    except subprocess.CalledProcessError as e:
        print(f"安装包时出错: {e}")
        raise

def add_installed_packages_to_sys_path(python_executable):
    """将pip安装的包目录添加到sys.path"""
    try:
        # 获取site-packages路径
        import site
        site_packages = site.getsitepackages()
        user_site = site.getusersitepackages()
        
        all_paths = site_packages + [user_site]
        for path in all_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)
                print(f"Added to sys.path: {path}")

    except Exception as e:
        print(f"添加安装目录到 sys.path 时出错: {e}")
        raise

def main():
    try:
        # 添加当前目录到 sys.path
        add_current_dir_to_sys_path()

        # 获取当前Python解释器路径
        python_executable = sys.executable
        print(f"使用的Python解释器: {python_executable}")

        # 安装依赖包
        install_packages(python_executable)

        # 将安装的包目录添加到 sys.path
        add_installed_packages_to_sys_path(python_executable)

    except Exception as e:
        print("安装过程失败，请检查错误信息。")
        traceback.print_exc()

if __name__ == "__main__":
    main()
