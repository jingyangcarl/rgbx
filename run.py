import shutil
import os
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gradio as gr


import torch
import torchvision
from diffusers import DDIMScheduler
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import imageio
import cv2

from rgb2x.load_image import load_exr_image, load_ldr_image
from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from x2rgb.pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
    
# Load pipeline
# pipe_rgb2x = StableDiffusionAOVMatEstPipeline.from_pretrained(
#     "zheng95z/rgb-to-x",
#     torch_dtype=torch.float16,
#     cache_dir=os.path.join("/home/jyang/projects/rgbx/rgb2x/model_cache"),
# ).to("cuda")
# pipe_rgb2x.scheduler = DDIMScheduler.from_config(
#     pipe_rgb2x.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
# )
# pipe_rgb2x.set_progress_bar_config(disable=True)
# pipe_rgb2x.to("cuda")


# Load pipeline
# pipe_x2rgb = StableDiffusionAOVDropoutPipeline.from_pretrained(
#     "zheng95z/x-to-rgb",
#     torch_dtype=torch.float16,
#     cache_dir=os.path.join("/home/jyang/projects/rgbx/x2rgb/model_cache"),
# ).to("cuda")
# pipe_x2rgb.scheduler = DDIMScheduler.from_config(
#     pipe_x2rgb.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
# )
# pipe_x2rgb.set_progress_bar_config(disable=True)
# pipe_x2rgb.to("cuda")

# Augmentation
def rgb2x(
    photo_path,
    seed = 2025,
    inference_step = 50,
    num_samples = 1,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if photo_path.endswith(".exr"):
        photo = load_exr_image(photo_path, tonemaping=True, clamp=True).to("cuda")
    elif (
        photo_path.endswith(".png")
        or photo_path.endswith(".jpg")
        or photo_path.endswith(".jpeg")
    ):
        photo = load_ldr_image(photo_path, from_srgb=True).to("cuda")
        
        # if resolution is over 1k, downsample to less than 1k
        if photo.shape[1] > 512: # photo in shape 3, H, W
            downsize = 512 / photo.shape[1]
            photo = torchvision.transforms.Resize((int(photo.shape[1] * downsize), int(photo.shape[2] * downsize)))(photo)

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    prompts = {
        "albedo": "Albedo (diffuse basecolor)",
        "normal": "Camera-space Normal",
        "roughness": "Roughness",
        "metallic": "Metallicness",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    return_list = []
    for i in tqdm(range(num_samples)):
        for aov_name in required_aovs:
            prompt = prompts[aov_name]
            generated_image = pipe_rgb2x(
                prompt=prompt,
                photo=photo,
                num_inference_steps=inference_step,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            generated_image = torchvision.transforms.Resize(
                (old_height, old_width)
            )(generated_image)

            generated_image = (generated_image, f"Generated {aov_name} {i}")
            return_list.append(generated_image)

    return return_list, prompts


def x2rgb(
    albedo_path,
    normal_path,
    roughness_path,
    metallic_path,
    irradiance_path,
    prompt,
    seed,
    inference_step,
    num_samples,
    guidance_scale,
    image_guidance_scale,
):
    if albedo_path is None:
        albedo_image = None
    elif albedo_path.endswith(".exr"):
        albedo_image = load_exr_image(albedo_path, clamp=True).to("cuda")
    elif (
        albedo_path.endswith(".png")
        or albedo_path.endswith(".jpg")
        or albedo_path.endswith(".jpeg")
    ):
        albedo_image = load_ldr_image(albedo_path, from_srgb=True).to("cuda")

    if normal_path is None:
        normal_image = None
    elif normal_path.endswith(".exr"):
        normal_image = load_exr_image(normal_path, normalize=True).to("cuda")
    elif (
        normal_path.endswith(".png")
        or normal_path.endswith(".jpg")
        or normal_path.endswith(".jpeg")
    ):
        normal_image = load_ldr_image(normal_path, normalize=True).to("cuda")

    if roughness_path is None:
        roughness_image = None
    elif roughness_path.endswith(".exr"):
        roughness_image = load_exr_image(roughness_path, clamp=True).to("cuda")
    elif (
        roughness_path.endswith(".png")
        or roughness_path.endswith(".jpg")
        or roughness_path.endswith(".jpeg")
    ):
        roughness_image = load_ldr_image(roughness_path, clamp=True).to("cuda")

    if metallic_path is None:
        metallic_image = None
    elif metallic_path.endswith(".exr"):
        metallic_image = load_exr_image(metallic_path, clamp=True).to("cuda")
    elif (
        metallic_path.endswith(".png")
        or metallic_path.endswith(".jpg")
        or metallic_path.endswith(".jpeg")
    ):
        metallic_image = load_ldr_image(metallic_path, clamp=True).to("cuda")

    if irradiance_path is None:
        irradiance_image = None
    elif irradiance_path.endswith(".exr"):
        irradiance_image = load_exr_image(
            irradiance_path, tonemaping=True, clamp=True
        ).to("cuda")
    elif (
        irradiance_path.endswith(".png")
        or irradiance_path.endswith(".jpg")
        or irradiance_path.endswith(".jpeg")
    ):
        irradiance_image = load_ldr_image(
            irradiance_path, from_srgb=True, clamp=True
        ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Set default height and width
    old_height, old_width = albedo_image.shape[1], albedo_image.shape[2]
    height = old_height // 8 * 8
    width = old_width // 8 * 8
    
    albedo_image = torchvision.transforms.Resize((height, width))(albedo_image)
    normal_image = torchvision.transforms.Resize((height, width))(normal_image)
    roughness_image = torchvision.transforms.Resize((height, width))(roughness_image)
    metallic_image = torchvision.transforms.Resize((height, width))(metallic_image)
    irradiance_image = torchvision.transforms.Resize((height, width))(irradiance_image)

    # Check if any of the input images are not None
    # and set the height and width accordingly
    images = [
        albedo_image,
        normal_image,
        roughness_image,
        metallic_image,
        irradiance_image,
    ]
    for img in images:
        if img is not None:
            height = img.shape[1]
            width = img.shape[2]
            break

    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    return_list = []
    for i in range(num_samples):
        generated_image = pipe_x2rgb(
            prompt=prompt,
            albedo=albedo_image,
            normal=normal_image,
            roughness=roughness_image,
            metallic=metallic_image,
            irradiance=irradiance_image,
            num_inference_steps=inference_step,
            height=height,
            width=width,
            generator=generator,
            required_aovs=required_aovs,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            guidance_rescale=0.7,
            # output_type="np",
        ).images[0]
        
        generated_image = torchvision.transforms.Resize(
            (old_height, old_width)
        )(generated_image)

        generated_image = (generated_image, f"Generated Image {i}")
        return_list.append(generated_image)

    if albedo_image is not None:
        albedo_image = albedo_image ** (1 / 2.2)
        albedo_image = albedo_image.cpu().numpy().transpose(1, 2, 0)
    else:
        albedo_image = np.zeros((height, width, 3))

    if normal_image is not None:
        normal_image = normal_image * 0.5 + 0.5
        normal_image = normal_image.cpu().numpy().transpose(1, 2, 0)
    else:
        normal_image = np.zeros((height, width, 3))

    if roughness_image is not None:
        roughness_image = roughness_image.cpu().numpy().transpose(1, 2, 0)
    else:
        roughness_image = np.zeros((height, width, 3))

    if metallic_image is not None:
        metallic_image = metallic_image.cpu().numpy().transpose(1, 2, 0)
    else:
        metallic_image = np.zeros((height, width, 3))

    if irradiance_image is not None:
        irradiance_image = irradiance_image ** (1 / 2.2)
        irradiance_image = irradiance_image.cpu().numpy().transpose(1, 2, 0)
    else:
        irradiance_image = np.zeros((height, width, 3))

    albedo_image = (albedo_image, "Albedo")
    normal_image = (normal_image, "Normal")
    roughness_image = (roughness_image, "Roughness")
    metallic_image = (metallic_image, "Metallic")
    irradiance_image = (irradiance_image, "Irradiance")

    return_list.append(albedo_image)
    return_list.append(normal_image)
    return_list.append(roughness_image)
    return_list.append(metallic_image)
    return_list.append(irradiance_image)

    return return_list


def render_olats(rets, camera, cam_downscale=8.0, resize_shape=None):
    
    def get_lightstage_camera(fpath, downscale=1.0):
        """
        Reads Lightstage camera parameters from file and optionally
        downscales intrinsics by a given factor.
        
        Args:
            fpath (str): Path to the Lightstage camera file.
            downscale (float): Downscaling factor, e.g., 2.0 halves the resolution.

        Returns:
            dict with keys:
                'Rt':  (3 x 4 or 4 x 4) extrinsic matrix
                'K':   (3 x 3) intrinsic matrix
                'fov': (3,) approximate field of view in degrees (for each dimension)
                'hwf': [height, width, focal_x] 
                'pp':  (2,) principal point [cx, cy]
        """
        # 1) Read lines from file
        with open(fpath) as f:
            txt = f.read().split('\n')
            
        # 2) Parse lines
        # Typically the text file has lines like:
        #   line 1: focal_x focal_y
        #   line 3: pp_x pp_y
        #   line 5: width height
        # Then lines 12..14: extrinsics
        focal = np.asarray(txt[1].split()).astype(np.float32)      # shape (2,)
        pp = np.asarray(txt[3].split()).astype(np.float32)         # shape (2,)
        resolution = np.asarray(txt[5].split()).astype(np.float32) # shape (2,) = [width, height]
        Rt = np.asarray([line.split() for line in txt[12:15]]).astype(np.float32)
        
        # 3) Compute field of view in radians, then convert to degrees
        #    fov_x = 2 * arctan( (width/2) / focal_x ), etc.
        fov = 2 * np.arctan(0.5 * resolution / focal)  # shape (2,)
        fov_deg = fov / np.pi * 180.0                  # convert to degrees
        
        # 4) If downscale != 1.0, adjust the camera intrinsics accordingly
        if downscale != 1.0:
            resolution = resolution / downscale
            focal = focal / downscale
            pp = pp / downscale
            # Recompute FOV if you want the scaled version
            # (If you keep ratio resolution/focal the same, angle stays the same,
            #  but let's re-derive it for completeness.)
            fov = 2 * np.arctan(0.5 * resolution / focal)
            fov_deg = fov / np.pi * 180.0
            
        if resize_shape is not None:
            H, W = resize_shape
            focal = focal * np.array([W, H]) / resolution
            pp = pp * np.array([W, H]) / resolution
            resolution = np.array([W, H])
            
                
        # 5) Compose the intrinsic matrix K
        # https://stackoverflow.com/questions/74749690/how-will-the-camera-intrinsics-change-if-an-image-is-cropped-resized
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal[0]
        K[1, 1] = focal[1]
        K[0, 2] = pp[0]
        K[1, 2] = pp[1]
        
        # 6) Return a dictionary of all camera parameters
        return {
            'Rt': Rt,                       # (3 x 4) or (4 x 4) extrinsic
            'K': K,                         # (3 x 3) intrinsic
            'fov': fov_deg,                # field of view in degrees
            'hwf': [resolution[1],         # height
                    resolution[0],         # width
                    focal[0]],             # focal_x (for NeRF-style notation)
            'pp': pp                       # principal point
        }
    
    cam_path = '/home/jyang/projects/ObjectReal/data/LightStageObjectDB/Redline/v1.2/v1.2_2/cameras'
    camid = camera[-2:]
    cam = get_lightstage_camera(os.path.join(cam_path, f'camera{camid}.txt'), downscale=cam_downscale)
    
    # build view_dirs
    H, W = cam['hwf'][:2]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cam['pp'][0])/cam['K'][0,0], -(j-cam['pp'][1])/cam['K'][1,1], -np.ones_like(i)], -1)
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6)

    # build lighting dirs
    olat_base = '/home/jyang/projects/ObjectReal/data/LSX'
    olat_pos_ = np.genfromtxt(f'{olat_base}/LSX3_light_positions.txt').astype(np.float32)
    olat_idx = np.genfromtxt(f'{olat_base}/LSX3_light_z_spiral.txt').astype(np.int32)
    r = R.from_euler('y', 180, degrees=True)
    olat_pos_ = (olat_pos_ @ r.as_matrix().T).astype(np.float32)
    omega_i_world = olat_pos_[olat_idx-1]
    
    # get material properties, Image to numpy
    albedo = np.asarray(rets[0][0]) / 255.0
    normal = np.asarray(rets[1][0]) / 255.0 * 2.0 - 1.0
    roughness = np.asarray(rets[2][0]) / 255.0
    metallic = np.asarray(rets[3][0]) / 255.0
    irradiance = np.asarray(rets[4][0]) / 255.0
    
    # ensure normal vectors are normalized
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
    
    # Define reflectance function (Cook-Torrance BRDF)
    def fresnel_schlick(cos_theta, F0):
        """Schlick's approximation for Fresnel term"""
        return F0 + (1 - F0) * np.power(1 - cos_theta, 5)

    def ggx_distribution(NdotH, alpha):
        """GGX normal distribution function (NDF)"""
        alpha2 = alpha * alpha
        denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0) ** 2
        return alpha2 / (np.pi * denom)

    def smith_schlick_ggx(NdotV, NdotL, alpha):
        """Smith Schlick-GGX Geometry function"""
        k = (alpha + 1) ** 2 / 8.0
        G1 = NdotV / (NdotV * (1 - k) + k)
        G2 = NdotL / (NdotL * (1 - k) + k)
        return G1 * G2

    def cook_torrance_brdf(N, V, L, albedo, roughness, metallic):
        """Cook-Torrance BRDF computation"""
        H = (V + L) / np.linalg.norm(V + L, axis=-1, keepdims=True)
        
        NdotL = np.maximum(np.sum(N * L, axis=-1, keepdims=True), 1e-6)
        NdotV = np.maximum(np.sum(N * V, axis=-1, keepdims=True), 1e-6)
        NdotH = np.maximum(np.sum(N * H, axis=-1, keepdims=True), 1e-6)
        VdotH = np.maximum(np.sum(V * H, axis=-1, keepdims=True), 1e-6)
        
        # F0 for metals and dielectrics
        F0 = 0.04 * (1 - metallic) + albedo * metallic
        F = fresnel_schlick(VdotH, F0)
        
        D = ggx_distribution(NdotH, roughness ** 2)
        G = smith_schlick_ggx(NdotV, NdotL, roughness ** 2)
        
        denominator = 4 * NdotV * NdotL + 1e-6
        specular = (D * F * G) / denominator
        
        k_s = F
        k_d = (1 - k_s) * (1 - metallic)
        
        diffuse = (albedo / np.pi) * k_d
        
        return (diffuse + specular) * NdotL, NdotL
    
    # def cook_torrance_brdf_torch(N, V, Ls, L_radiance, albedo, roughness, metallic, device="cuda"):
    def cook_torrance_brdf_torch(N, V, Ls, L_radiance, albedo, roughness, metallic, device="cuda"):
        """
        Vectorized Cook-Torrance BRDF with HDRI lighting, optimized for PyTorch GPU.

        - Automatically converts NumPy inputs to PyTorch.
        - Runs on CUDA (if available) and returns the result as a NumPy array.

        Inputs:
        - N: (H, W, 3)  -> Normal map
        - V: (H, W, 3)  -> View direction
        - Ls: (num_lights, 3)  -> Light directions
        - L_radiance: (num_lights, 3)  -> HDRI radiance per light
        - albedo: (H, W, 3)  -> Surface albedo
        - roughness: (H, W, 3)  -> Roughness map
        - metallic: (H, W, 3)  -> Metallic map

        Returns:
        - Final rendered image (H, W, 3) as a NumPy array
        """

        # Determine device (use GPU if available)
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

        # Convert NumPy arrays to PyTorch tensors and move to GPU
        N = torch.tensor(N, dtype=torch.float32, device=device)
        V = torch.tensor(V, dtype=torch.float32, device=device)
        Ls = torch.tensor(Ls, dtype=torch.float32, device=device)  # (num_lights, 3)
        L_radiance = torch.tensor(L_radiance, dtype=torch.float32, device=device)  # (num_lights, 3)
        albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
        roughness = torch.tensor(roughness, dtype=torch.float32, device=device)
        metallic = torch.tensor(metallic, dtype=torch.float32, device=device)

        num_lights = Ls.shape[0]

        # Expand dimensions for batch processing
        N_exp = N.unsqueeze(0)  # (1, H, W, 3)
        V_exp = V.unsqueeze(0)  # (1, H, W, 3)
        L_exp = Ls.view(num_lights, 1, 1, 3)  # (num_lights, 1, 1, 3)

        # Compute halfway vectors
        H = (V_exp + L_exp)
        H = H / (torch.norm(H, dim=-1, keepdim=True) + 1e-6)  # Normalize

        # Compute dot products
        NdotL = torch.clamp(torch.sum(N_exp * L_exp, dim=-1, keepdim=True), min=1e-4)
        NdotV = torch.clamp(torch.sum(N_exp * V_exp, dim=-1, keepdim=True), min=1e-4)
        NdotH = torch.clamp(torch.sum(N_exp * H, dim=-1, keepdim=True), min=1e-4)
        VdotH = torch.clamp(torch.sum(V_exp * H, dim=-1, keepdim=True), min=1e-4)

        # Fresnel Schlick approximation
        F0 = (0.04 * (1 - metallic) + albedo * metallic)[None,...]  # ✅ No need to expand
        F = F0 + (1 - F0) * torch.pow(1 - VdotH, 5)

        # GGX Normal Distribution Function (NDF)
        alpha2 = roughness ** 2
        D = alpha2 / (torch.pi * ((NdotH ** 2 * (alpha2 - 1) + 1) ** 2) + 1e-6)

        # Smith Schlick-GGX Geometry function
        k = (roughness + 1) ** 2 / 8.0
        G1 = NdotV / (NdotV * (1 - k) + k)
        G2 = NdotL / (NdotL * (1 - k) + k)
        G = G1 * G2

        # Specular term
        denominator = 4 * NdotV * NdotL + 1e-6
        specular = (D * F * G) / denominator

        # Diffuse term (Lambertian)
        k_s = F
        k_d = (1 - k_s) * (1 - metallic)
        diffuse = (albedo / torch.pi) * k_d

        # BRDF per light
        brdf_per_light = (diffuse + specular) * NdotL  # (num_lights, H, W, 3)

        # Sum over all lights, applying HDRI radiance
        final_render = torch.sum(brdf_per_light * L_radiance.view(num_lights, 1, 1, 3), dim=0)  # (H, W, 3)
        final_ndotl = torch.sum(NdotL * L_radiance.view(num_lights, 1, 1, 3), dim=0)  # (H, W, 3)

        # Convert back to NumPy for output
        return brdf_per_light.cpu().numpy(), NdotL.repeat(1,1,1,3).cpu().numpy(), final_render.cpu().numpy(), final_ndotl.cpu().numpy()

    def hdri_to_direction(hdri, H, W):
        """Generate a corresponding direction map for the HDRI"""
        theta = np.linspace(0, np.pi, H)  # Vertical angle (0=top, pi=bottom)
        phi = np.linspace(-np.pi, np.pi, W)  # Horizontal angle (-pi to pi)

        theta, phi = np.meshgrid(theta, phi, indexing='ij')

        # Convert to Cartesian coordinates (world space)
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)  # Up direction
        z = np.sin(theta) * np.sin(phi)

        directions = np.stack([x, y, z], axis=-1)
        return directions

    # Compute shading
    V = -dirs  # View direction (negate view vectors)
    
    rendered_olat = []
    rendered_ndotl = []
    # for i in tqdm(range(omega_i_world.shape[0])):
    #     L = omega_i_world[i]  # Light direction
    #     L = L / np.linalg.norm(L)  # Normalize
        
    #     # Compute Cook-Torrance shading
    #     shading, ndotl = cook_torrance_brdf(normal, V, L, albedo, roughness, metallic)
        
    #     # Apply irradiance
    #     exposure = 40
    #     shaded_image = shading * irradiance * exposure
        
    #     rendered_olat.append(shaded_image)
    #     rendered_ndotl.append(ndotl)
    
    
    Ls = omega_i_world / np.linalg.norm(omega_i_world, axis=-1, keepdims=True)
    shadings, ndotls, _, _ = cook_torrance_brdf_torch(normal, V, Ls, np.ones_like(Ls), albedo, roughness, metallic)
    for i in range(shadings.shape[0]):
        exposure = 40
        rendered_olat.append(shadings[i] * irradiance * exposure)
        rendered_ndotl.append(ndotls[i])
    
    
    hdris = os.scandir('/labworking/Users_A-L/jyang/data/lightProbe/general/exr_downsample_20_40')
    for hdri in tqdm(list(hdris)[:20]):
        hdri = imageio.imread(hdri.path)
        # downsample hdri to 15 * 30 to avoid OOM
        hdri = cv2.resize(hdri, (15, 30))
        Ls = hdri_to_direction(hdri, hdri.shape[0], hdri.shape[1])
        Ls = Ls.reshape(-1, 3)
        Ls_ir = hdri.reshape(-1, 3)
        
        _, _, shadings_sum, ndotl_sum = cook_torrance_brdf_torch(normal, V, Ls, Ls_ir, albedo, roughness, metallic)
        torch.cuda.empty_cache()
            
        rendered_olat.append(shadings_sum / Ls.shape[0] * exposure)
        rendered_ndotl.append(ndotl_sum)
    
    return rendered_olat, rendered_ndotl
    
    return rendered_olat, rendered_ndotl

def copy_realworld_and_run(db_src, db_dst):
    
    imgs = os.listdir(db_src)
    for img in imgs:
        
        if 'md' in img:
            continue
        
        img_name = img.split('.')[0]
        src = os.path.join(db_src, img)
        dst = os.path.join(db_dst, img_name, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # print(f'Copying {src} to {dst}')
        shutil.copy(src, dst)
        
        # run rgbx pipeline
        test_folder = os.path.join(db_dst, img_name)
        try:
                
            # run the pipeline when enabled, otherwise, load the images and run the render_olats
            if not os.path.exists(os.path.join(test_folder, 'Generated_albedo_0.png')):
                
                print(f'Running rgbx pipeline for {dst}')
                rets, prompts = rgb2x(dst)
                
                # save out the results
                x_paths = {}
                for (ret, prompt) in zip(rets, prompts):
                    img, name = ret
                    name = name.replace(' ', '_')
                    x_path = os.path.join(test_folder, f'{name}.png')
                    img.save(x_path)
                    x_paths[prompt] = x_path
                
                # 
                rets = x2rgb(
                    albedo_path=x_paths['albedo'],
                    normal_path=x_paths['normal'],
                    roughness_path=x_paths['roughness'],
                    metallic_path=x_paths['metallic'],
                    irradiance_path=x_paths['irradiance'],
                    prompt='',
                    seed=2025,
                    inference_step=50,
                    num_samples=1,
                    guidance_scale=7.5,
                    image_guidance_scale=1.5,
                )
                
                rets[0][0].save(os.path.join(test_folder, f'mixed_w2_{rets[0][1]}.png'))
            
            print(f'Loading rgbx pipeline results for {test_folder}')
            rets = []
            for prompt in ['albedo', 'normal', 'roughness', 'metallic', 'irradiance']:
                x_path = os.path.join(test_folder, f'Generated_{prompt}_0.png')
                
                if prompt == 'metallic':
                    rets.append((imageio.imread(x_path) / 10., prompt))
                else:
                    rets.append((imageio.imread(x_path), prompt))
            
            camera = 'cam07' # hardcode the camera for realworld since 07 is the center
            olats, ndotls = render_olats(rets, camera, resize_shape=rets[0][0].shape[:2]) # all assets in 0-255
            
            # save olats in a folder and as a video
            olats_folder = os.path.join(test_folder, 'olats')
            os.makedirs(olats_folder, exist_ok=True)
            for i, olat in enumerate(olats):
                olat_path = os.path.join(olats_folder, f'olat_{i:03d}.png')
                olat = (np.clip(olat, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(olat_path, olat)
                
            # create a video from the olats
            imageio.mimwrite(os.path.join(test_folder, 'olats.mp4'), olats, fps=30, quality=8)
            imageio.mimwrite(os.path.join(test_folder, 'ndotls.mp4'), ndotls, fps=30, quality=8)
            
            # read olats.mp4 and save each frame
            olats_video = imageio.get_reader(os.path.join(test_folder, 'olats.mp4'))
            olats_folder = os.path.join(test_folder, 'olats_videoframes')
            os.makedirs(olats_folder, exist_ok=True)
            for i, frame in enumerate(olats_video):
                olat_path = os.path.join(olats_folder, f'olat_{i:03d}.png')
                imageio.imwrite(olat_path, frame)
            olats_video.close()
            
        except Exception as e:
            print(f'Error running iid pipeline for: {e}')
            continue
            
            


def copy_LightStageObjectDB_and_run(db_src, db_dst, db_dst_root, obj_list):
    
    objects = os.listdir(db_src)
    for obj in objects:
        
        if 'glove/static/cam07' not in db_dst:
            continue
        
        if obj == 'cameras':
            continue # skip cameras folder
        
        if obj_list and obj not in obj_list:
            continue # skip objects not in obj_list
        
        pattern = 'static'
        
        cameras = os.listdir(os.path.join(db_src, obj, pattern))
        for camera in cameras:
            
            # copy mixed_w2.jpg
            src = os.path.join(db_src, obj, pattern, camera, 'mixed_w2.jpg')
            dst = os.path.join(db_dst, obj, pattern, camera, 'mixed_w2.jpg')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # print(f'Copying {src} to {dst}')
            shutil.copy(src, dst)
            
            # run rgbx pipeline
            test_folder = os.path.join(db_dst, obj, pattern, camera)
            test_folder_relative = os.path.relpath(test_folder, db_dst_root)
            try:
                    
                # run the pipeline when enabled, otherwise, load the images and run the render_olats
                if not os.path.exists(os.path.join(test_folder, 'Generated_albedo_0.png')):
                    
                    print(f'Running rgbx pipeline for {dst}')
                    rets, prompts = rgb2x(dst)
                    
                    # save out the results
                    x_paths = {}
                    for (ret, prompt) in zip(rets, prompts):
                        img, name = ret
                        name = name.replace(' ', '_')
                        x_path = os.path.join(test_folder, f'{name}.png')
                        img.save(x_path)
                        x_paths[prompt] = x_path
                    
                    # 
                    rets = x2rgb(
                        albedo_path=x_paths['albedo'],
                        normal_path=x_paths['normal'],
                        roughness_path=x_paths['roughness'],
                        metallic_path=x_paths['metallic'],
                        irradiance_path=x_paths['irradiance'],
                        prompt='',
                        seed=2025,
                        inference_step=50,
                        num_samples=1,
                        guidance_scale=7.5,
                        image_guidance_scale=1.5,
                    )
                    
                    rets[0][0].save(os.path.join(test_folder, f'mixed_w2_{rets[0][1]}.png'))
                
                # else:
                print(f'Loading rgbx pipeline results for {test_folder}')
                rets = []
                for prompt in ['albedo', 'normal', 'roughness', 'metallic', 'irradiance']:
                    x_path = os.path.join(test_folder, f'Generated_{prompt}_0.png')
                    rets.append((imageio.imread(x_path), prompt))
                
                olats, ndotls = render_olats(rets, camera)
                
                # save olats in a folder and as a video
                olats_folder = os.path.join(test_folder, 'olats')
                os.makedirs(olats_folder, exist_ok=True)
                for i, olat in enumerate(olats):
                    olat_path = os.path.join(olats_folder, f'olat_{i:03d}.png')
                    olat = (np.clip(olat, 0, 1) * 255).astype(np.uint8)
                    imageio.imwrite(olat_path, olat)
                    
                # create a video from the olats
                imageio.mimwrite(os.path.join(test_folder, 'olats.mp4'), olats, fps=30, quality=8)
                imageio.mimwrite(os.path.join(test_folder, 'ndotls.mp4'), ndotls, fps=30, quality=8)
                
                # read olats.mp4 and save each frame
                olats_video = imageio.get_reader(os.path.join(test_folder, 'olats.mp4'))
                olats_folder = os.path.join(test_folder, 'olats_videoframes')
                os.makedirs(olats_folder, exist_ok=True)
                for i, frame in enumerate(olats_video):
                    olat_path = os.path.join(olats_folder, f'olat_{i:03d}.png')
                    imageio.imwrite(olat_path, frame)
                olats_video.close()
                
                exit()
                
            except Exception as e:
                print(f'Error running iid pipeline for {test_folder_relative}: {e}')
                continue
            

if __name__ == '__main__':
    
    v, res = 'v1.2', 8
    db_src_root = '/labworking/Users_A-L/jyang/data'
    db_dst_root = '/home/jyang/projects/rgbx/data'
    db_src = db_src_root + f'/LightStageObjectDB/Redline/jpg/{v}/{v}_{res}'
    db_dst = db_dst_root + f'/LightStageObjectDB/Redline/jpg/{v}/{v}_{res}'
    
    obj_list = []

    # disable the openexternal when running the code
    # Press Ctrl + Shift + P and type: Preferences: Open Settings (JSON)
    # Add the following entry: "window.openExternal": false
    
    # copy_LightStageObjectDB_and_run(db_src, db_dst, db_dst_root, obj_list)
    
    db_src = '/home/jyang/projects/ObjectReal/data/realworld'
    db_dst = '/home/jyang/projects/rgbx/data/realworld'
    copy_realworld_and_run(db_src, db_dst)
    
    
    