import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import mne_connectivity
from scipy.stats import ttest_ind

sublist = ["sub-01c","sub-02c","sub-03c","sub-04c","sub-05c","sub-06c","sub-07c","sub-08c","sub-09c","sub-10c","sub-11c","sub-12c","sub-13c","sub-14c","sub-15c","sub-16c","sub-18c","sub-19c","sub-20c","sub-21c","sub-22c","sub-23c","sub-24c","sub-01e","sub-02e","sub-03e","sub-04e","sub-05e","sub-06e","sub-07e","sub-08e","sub-09e","sub-10e","sub-11e","sub-12e","sub-13e","sub-14e","sub-15e","sub-16e","sub-17e","sub-18e","sub-19e","sub-20e","sub-21e","sub-22e","sub-23e","sub-24e","sub-25e","sub-26e","sub-27e","sub-28e","sub-29e","sub-31e","sub-33e","sub-34e","sub-36e","sub-37e","sub-38e","sub-39e","sub-40e","sub-41e","sub-42e","sub-43e"]
all_theta = []
all_alpha = []
all_beta = []
bands = {"theta": (4,8), "alpha": (8,12),"beta": (12,30)}
all_age = []
all_sex = []
all_coh = []

for fullname in sublist:

  print(fullname)

  metad=pd.read_excel("NeuroTechs Dataset for Stem Skills/extra_metadata.xlsx", sheet_name="Individual metadata",index_col=0)
  dir = "".join((fullname,"/programming_responses.csv"))
  direeg="".join((fullname,"/ses-1/eeg/",fullname,"_ses-1_task-STEMSKILLS_eeg.set"))
  dirchan="".join((fullname,"/ses-1/eeg/",fullname,"_ses-1_electrodes.tsv"))
  age = metad.loc[fullname, 'Age']
  sex = metad.loc[fullname, 'AAB Sex']
  flexdir = "".join(("NeuroTechs Dataset for Stem Skills/",fullname,"/programming_responses.csv"))
  sub_dir = "".join(("NeuroTechs Dataset for Stem Skills/",fullname,"/"))

  events=pd.read_csv(flexdir,index_col=0)

  flexdir2 = "".join(("NeuroTechs Dataset for Stem Skills/",fullname,"/ses-1/eeg/",fullname,"_ses-1_task-STEMSKILLS_eeg.set"))
  flexdir3 = "".join(("NeuroTechs Dataset for Stem Skills/",fullname,"/ses-1/eeg/",fullname,"_ses-1_electrodes.tsv"))

  markers = events.iloc[:,[1]]
  question_appearance = markers - 3

  eeg_file=mne.io.read_raw_eeglab(flexdir2, preload=True)

  # Datos
  fs = 250

  # Asignar montaje
  mne.channels.read_custom_montage(flexdir3)

  # Re-referenciar
  eeg_file.set_eeg_reference('average')

  # Filtro en bandas de interés (4hz - 30hz)
  eeg_file.filter(4, 30)

  # Crear matriz de eventos, decartamos las preguntas que no cumplan con el tiempo mínimo de la época
  diffs = np.diff(question_appearance)
  valid_mask = diffs >= 2
  filtered_times = question_appearance[valid_mask]
  event_samples = (filtered_times * fs).astype(int)

  events = np.column_stack((
      event_samples,
      np.zeros(len(event_samples), dtype=int),
      np.ones(len(event_samples), dtype=int)
  ))

  # Crear épocas (-1.5s, 2.5s), los que no cumplan los decartamos
  epochs = mne.Epochs(
      eeg_file,
      events,
      event_id=1,
      tmin=-1,
      tmax=2
  )

  # Calcular la coherencia
  freqs = np.arange(4, 31, 0.1)   # 4 a 30 Hz
  n_cycles = freqs / 2         # equilibrio tiempo-frecuencia

  con = mne_connectivity.spectral_connectivity_epochs(
      epochs,
      method='imcoh',
      mode='cwt_morlet',
      sfreq=fs,
      cwt_freqs=freqs,
      cwt_n_cycles=n_cycles,
      fmin=4,
      fmax=30,
      tmin=-1.0
  )
  con_data = con.get_data()
  con_dense= con.get_data(output='dense')

  # Evitar valores exactamente ±1
  con_dense = np.clip(con_dense, -0.999999, 0.999999)

  # Fisher Z transform
  con_dense_z = np.arctanh(con_dense)

  all_coh.append(con_dense_z)
  all_age.append(age)
  all_sex.append(sex)


all_age = np.array(all_age)
all_sex = np.array(all_sex)




all_coh_stacked = np.array(all_coh)

male_idx = all_sex == "Male"
female_idx = all_sex == "Female"

all_male_coh = all_coh_stacked[male_idx]
all_female_coh = all_coh_stacked[female_idx]

tvals, pvals = ttest_ind(all_male_coh, all_female_coh, axis=0, equal_var=False)

pvals = np.nan_to_num(pvals, nan=1)
alpha = 0.01
sig_mask = pvals < alpha


channellist = ["Fz","C3","Cz","C4","Pz","PO7","Oz","PO8"]

t_min = -1
t_max = 2

Ccohm = np.array(all_male_coh).mean(axis=0)
Ccohf=np.array(all_female_coh).mean(axis=0)
Ccoh = np.array(all_coh).mean(axis=0)
times = np.linspace(t_min, t_max, Ccoh.shape[3])
freqs = np.linspace(4, 30, Ccoh.shape[2])

for i in range(len(channellist)):
    for j in range(len(channellist)):

        # Create 1 figure with 3 subplots side-by-side
        # figsize=(width, height) - you can adjust this to fit your screen/needs
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # --- 1. All Data Plot ---
        im0 = axes[0].imshow(
            Ccoh[i,j,:,:],
            cmap='bwr',
            origin='lower',
            extent=[t_min, t_max, 4, 30],
            aspect=((3/Ccoh.shape[3])/(26/Ccoh.shape[2]))
        )
        axes[0].set_title(f"{channellist[i]} to {channellist[j]} (All)")
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_xlabel("Time")

        # Scatter plot for significant points
        fy, fx = np.where(sig_mask[i,j,:,:])
        axes[0].scatter(times[fx], freqs[fy], s=5, color="cyan")

        # --- 2. Male Data Plot ---
        im1 = axes[1].imshow(
            Ccohm[i,j,:,:],
            cmap='bwr',
            origin='lower',
            extent=[t_min, t_max, 4, 30],
            aspect=((3/Ccoh.shape[3])/(26/Ccoh.shape[2]))
        )
        axes[1].set_title(f"{channellist[i]} to {channellist[j]} (Male)")
        axes[1].set_xlabel("Time")
        # Optional: axes[1].set_yticks([]) if you want to hide the Y-axis numbers on middle plots

        # --- 3. Female Data Plot ---
        im2 = axes[2].imshow(
            Ccohf[i,j,:,:],
            cmap='bwr',
            origin='lower',
            extent=[t_min, t_max, 4, 30],
            aspect=((3/Ccoh.shape[3])/(26/Ccoh.shape[2]))
        )
        axes[2].set_title(f"{channellist[i]} to {channellist[j]} (Female)")
        axes[2].set_xlabel("Time")

        # --- Shared Colorbar ---
        # By passing ax=axes (the array of all 3 subplots), matplotlib automatically
        # sizes the colorbar to span the height of the plots on the right side.
        fig.colorbar(im2, ax=axes, label="Coherence")
        im0.set_clim(-0.2,0.2)
        im1.set_clim(-0.2, 0.2)
        im2.set_clim(-0.2, 0.2)

        # Save and close the combined figure
        filename = f"{channellist[i]}_{channellist[j]}_combined_0.05.png"
        plt.savefig(filename, dpi=300) # Added dpi=300 for higher quality publications
        plt.close(fig)