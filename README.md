# Nerf_3d_reconstruction

# **Problem statement**

The problem at hand focuses on how scenes are represented for view synthesis and rendering in computer graphics. The challenge lies in accurately reconstructing complex scenes with high-resolution geometry and appearance using neural networks. This is a significant problem because traditional 3D reconstruction methods often rely on discrete representations like voxel grids or triangle meshes, which struggle to capture the fine details and intricate geometry of real-world scenes.

The NeRF (Neural Radiance Fields) algorithm seeks to address these limitations by introducing a new approach that represents scenes as continuous neural radiance fields. NeRF uses a fully connected deep network to map 5D coordinates to volume density and view-dependent emitted radiance, allowing for the synthesis of photorealistic novel views of scenes with complex geometry and appearance.

This innovative method not only improves the fidelity of scene reconstruction but also offers a more efficient and effective way to optimize scene representations for rendering realistic images. Therefore, researching how to enhance scene representation through neural radiance fields is essential for advancing computer graphics, enabling the creation of visually compelling and detailed virtual environments that closely resemble real-world scenes.

# **Introduction**


# **Usage**

Clone the Repository

```bash
git clone https://github.com/your-username/nerf-pipeline.git
cd nerf-pipeline
```
Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the Requirements

```bash
pip install -r requirements.txt
```

Dataset 
```bash
https://drive.google.com/drive/folders/18bwm-RiHETRCS5yD9G00seFIcrJHIvD-
```

Run the script
```bash
python main.py

```
File Structure


- main.py: The entry point of the application.
- model.py: Contains the NerfModel class.
- losses.py: Contains the loss functions (mse, psnr, ssim).
- utils.py: Contains utility functions (compute_accumulated_transmittance, render_rays).
- train.py: Contains the training and testing functions.

