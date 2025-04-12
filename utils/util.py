import numpy as np
import matplotlib.pyplot as plt

def visualize_results(reals_A, fakes_A, recs_A, reals_B, fakes_B, recs_B):
    reals_A = reals_A.cpu().numpy()
    fakes_A = fakes_A.cpu().numpy()
    recs_A = recs_A.cpu().numpy()
    reals_B = reals_B.cpu().numpy()
    fakes_B = fakes_B.cpu().numpy()
    recs_B = recs_B.cpu().numpy()
    
    fig, axes = plt.subplots(16, 6, figsize=(16, 30))
    axes = axes.flatten()
    for i in range(16):
        real_A = np.transpose(reals_A[i], (1, 2, 0))
        fake_A = np.transpose(fakes_A[i], (1, 2, 0))
        rec_A = np.transpose(recs_A[i], (1, 2, 0))
        real_B = np.transpose(reals_B[i], (1, 2, 0))
        fake_B = np.transpose(fakes_B[i], (1, 2, 0))
        rec_B = np.transpose(recs_B[i], (1, 2, 0))

        # Denormalization
        real_A = (real_A * 0.5) + 0.5
        real_B = (real_B * 0.5) + 0.5
        
        axes[6*i].imshow(real_A)
        axes[6*i].set_title("real A")
        axes[6*i].axis('off')

        axes[6*i+1].imshow(fake_A)
        axes[6*i+1].set_title("fake A")
        axes[6*i+1].axis('off')
        
        axes[6*i+2].imshow(rec_A)
        axes[6*i+2].set_title("rec A")
        axes[6*i+2].axis('off')

        axes[6*i+3].imshow(real_B)
        axes[6*i+3].set_title("real B")
        axes[6*i+3].axis('off')
        
        axes[6*i+4].imshow(fake_B)
        axes[6*i+4].set_title("fake B")
        axes[6*i+4].axis('off')
        
        axes[6*i+5].imshow(rec_B)
        axes[6*i+5].set_title("rec B")
        axes[6*i+5].axis('off')
    

    plt.tight_layout()
    plt.show()
