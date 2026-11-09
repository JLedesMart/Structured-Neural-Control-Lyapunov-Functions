# Structured-Neural-Control-Lyapunov-Functions
This repo contains the code for the paper "Learning and Verifying Stabilizing Controllers with Structured Neural
Control Lyapunov Functions" by Jose Luis Ledesma Martin and Patricia Pauli. 

## Requirements
The following must be installed manually:
- Python 3.11
- CPLEX 22.2.0.0
- auto-lirpa 0.7.2, installed from <https://github.com/Verified-Intelligence/auto_LiRPA>.

The remaining packages are installed via:
```
pip install -r requirements.txt
```
## Usage
The repo has one folder per dynamical system (inverted pendulum and path tracking), and each on contains a folder per Lyapunov architecture (ICNN-euclidean2, ICNN-L1, MLP, MLP-ReLU-L1, PLNet, and Quadratic).
To run an experiment, choose the system and Lyapunov architecture, open the .py file in its folder and change the seed variable at the top to the seed you want. Then run the script, e.g.:
```
python InvertedPendulum/PLNet/InvertedPendulumPLNet.py
```
## Results
The results are saved in the same folder containing the script of the experiment.

## Acknowledgements
Parts of this code was taken or built from <https://github.com/jlwu002/nlc_discrete/tree/main>, the repo for the paper "Neural Lyapunov Control for Discrete-Time Systems" by Junlin Wu, Andrew Clark, Yiannis Kantaros, Yevgeniy Vorobeychik.

Claude (Anthropic) and Gemini (Google) were used to support code development and debugging. The authors reviewed and verified all AI-assisted contributions and take full responsibility for the scientific content and results.

## License
With the exception of the cells of code marked with `# %% adapted from nlc_discrete`, `# %% taken from nlc_discrete` or similar, the code is released under the [MIT License](LICENSE). The cells marked with `# %% adapted from nlc_discrete` or `# %% taken from nlc_discrete` are adapted or taken from <https://github.com/jlwu002/nlc_discrete/tree/main>, which are used with the original authors' permission, and remain subject to their terms. Please contact them before using those parts.
