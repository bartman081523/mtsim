# Arbeitsordner
mkdir mt_hybrid && cd mt_hybrid

# Repos klonen
git clone https://github.com/manulera/simulationsLeraRamirez2021.git   external/lera_spindle
git clone https://github.com/WilliamLec/kMC_MoTub.git                  external/kmc_motub
git clone https://github.com/virtualcell/vcell.git                     external/vcell
git clone https://github.com/ngudimchuk/Four_State_MT_model.git        external/four_state_mt   # MATLAB
git clone https://github.com/TheonlyqueenAC/Microtubule_Simulation.git external/mt_quantum

cd ..
# Python-Env
python -m venv .venv-mtsim && source .venv-mtsim/bin/activate
pip install -r requirements.txt
