from multiprocessing import cpu_count
from codetrace.interp_utils import *
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

exp_dir = "/home/franlucc/projects/codetrace/codetrace/type_inf_exp"
with open(f"{exp_dir}/exp_data/v3/steering_res.json", "r") as f:
    ds = json.load(f)
    
df = pd.DataFrame(ds)
# incorrect_df = df[df['success'] == False]
incorrect_df = df
print(incorrect_df)

# filter out len > 8000
incorrect_df = incorrect_df[incorrect_df['prompt'].str.len() < 8000]
# sample some amount
incorrect_df = incorrect_df.sample(25, random_state=24)
print(incorrect_df[incorrect_df['success'] == True]['label'].value_counts())

model = LanguageModel("/home/arjun/models/starcoderbase-1b", device_map="cuda")
activations : torch.tensor = collect_hidden_states_at_tokens(model, 
                                              incorrect_df['prompt'].tolist(), 
                                              STARCODER_FIM.token,
                                              layers = [14])
add_steering_vec = False

if add_steering_vec:
    # add steering vector
    steering_vector = torch.load(f"{exp_dir}/exp_data/v3/diff_tensor.pt")
    print(steering_vector.shape)
    activations = activations + steering_vector[14, -1,:]

print(activations.shape)
# # tsne projection, see if there is a separating hyperplane
tsne_model = TSNE(n_components=2, 
                  perplexity = 10, # between 5-10, default 30, < num_samples
                  early_exaggeration=12.0, # default 12.0
                  metric='cosine', # default euclidean, maybe cosine?
                  verbose=1,
                  n_jobs = cpu_count(),
                #   n_iter=5000,
                #   n_iter_without_progress=500,
                  random_state=0)

embedded_in = activations[0, :, :,:].squeeze().detach().cpu().numpy()
print(embedded_in.shape)
embedded_out = tsne_model.fit_transform(embedded_in)
print(embedded_out.shape)


type_colors, colors, typ_list = [], [], []
for ex in datasets.Dataset.from_pandas(incorrect_df):
    if ex["success"]:
        colors.append("green")
    else:
        colors.append("red")
    
    typ_list.append(ex["label"])
    
    
plt.scatter(embedded_out[:, 0], embedded_out[:, 1], c=colors)

# make annotations for each x,y
for i,lbl in enumerate(typ_list):
    plt.annotate(lbl, (embedded_out[i, 0], embedded_out[i, 1]))
    
plt.tight_layout()
steer = "steer" if add_steering_vec else ""
plt.savefig(f"{exp_dir}/exp_data/v3/tsne_{steer}_pp{tsne_model.perplexity}_ee{tsne_model.early_exaggeration}_{tsne_model.metric}.png")
