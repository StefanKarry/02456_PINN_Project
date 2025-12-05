import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# paths 
working_dir = os.getcwd()
weights_path = 'lib/weights/PINNModel_Sphere.pth'
results_dir  = os.path.join(working_dir, "lib/results")
os.makedirs(results_dir, exist_ok=True)
current_dir = os.getcwd()
log_dir = os.path.join(current_dir, 'lib/logs')

images_dir = os.path.join(working_dir, "images")

#import model
from lib.model.PINNs import PINNModel_Sphere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grid
# ----------
N_theta = 400
N_phi   = 200
N_t     = 50

theta_star = np.linspace(0, np.pi,     N_theta)
phi_star   = np.linspace(0, 2*np.pi,   N_phi)
t_star     = np.linspace(0, 2.0,       N_t)

Theta, Phi, T = np.meshgrid(theta_star, phi_star, t_star, indexing="ij")
X_star = np.stack([Theta.ravel(), Phi.ravel(), T.ravel()], axis=1)
X_tensor = torch.tensor(X_star, dtype=torch.float32).to(device)


# load model & predict
# -----------------------
model = PINNModel_Sphere().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
with torch.no_grad():
    u_pred = model(X_tensor).cpu().numpy().flatten()

# Gem prediction 
#torch.save(torch.tensor(u_pred), os.path.join(results_dir, "predict_dec.pt"))
#print("Saved tensor to lib/results/prediction_dec.pt")

# 2D-plot of time t=1.2
# ---------------------
t_plot=float(1.2)
# Find nærmeste tid
t_idx = np.argmin(np.abs(t_star - t_plot))
closest_t = t_star[t_idx]
print(f"Plotting t = {closest_t:.4f}")
theta_slice = Theta[:,:,t_idx].flatten()
phi_slice   = Phi[:,:,t_idx].flatten()
u_slice     = u_pred.reshape(N_theta, N_phi, N_t)[:,:,t_idx].flatten()


fig, ax = plt.subplots(1,2, figsize = (12, 5))
im1 = ax[0].pcolormesh(phi_slice.reshape(N_theta, N_phi), theta_slice.reshape(N_theta, N_phi),
                       u_slice.reshape(N_theta, N_phi), cmap="turbo", shading='auto')
plt.colorbar(im1, ax=ax[0], label=f"$u(\\theta, \\phi, t={closest_t:.3f})$")
ax[0].set_xlabel("$\\phi$")
ax[0].set_ylabel("$\\theta$")
ax[0].set_title(f"PINN Prediction at t={closest_t:.3f}")


# # save fig
# save_path = os.path.join(results_dir, f"u3D_pred_2D_t{closest_t:.3f}.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"Figur gemt: {save_path}")

# creating error map
U_real = np.load(f'wave_sphere_exact_1.184.npz')['U']
error_map = np.abs(U_real - u_slice.reshape(N_theta, N_phi))

im2 = ax[1].pcolormesh(phi_slice.reshape(N_theta, N_phi), theta_slice.reshape(N_theta, N_phi),
                       error_map, cmap="inferno", shading='auto')
plt.colorbar(im2, ax=ax[1], label="Absolute Error")
ax[1].set_xlabel("$\\phi$")
ax[1].set_ylabel("$\\theta$")
ax[1].set_title(f"Absolute Error at t={closest_t:.3f}")

# # save error map figure
# error_map_path = os.path.join(results_dir, f"error_map_t{closest_t:.3f}.png")
# plt.savefig(error_map_path, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"Error map saved: {error_map_path}")

#saving combined figure
combined_path = os.path.join(images_dir, f"u3D_pred_and_error_2D_t{closest_t:.3f}.png")
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Combined figure saved: {combined_path}")


# saving loss history (as log(loss))
train_hist = np.loadtxt(os.path.join(log_dir, "training_loss.txt"), delimiter=",")
loss_hist = train_hist[:,1]
epochs = train_hist[:,0]

plt.figure(figsize=(7,5))
plt.plot(epochs, np.log(loss_hist))
plt.xlabel("Epoch")
plt.ylabel("log(Loss)")
plt.title("Training Loss History")
plt.grid(linestyle = '--')
plt.savefig(os.path.join(images_dir, "training_loss_history.png"), dpi=300)
plt.close()
