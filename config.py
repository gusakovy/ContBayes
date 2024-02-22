import os

# directory  definitions
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'contbayes')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# config variables
MODULATION_TYPE = "QPSK"
H_COEF = 0.8
LINEAR_CHANNEL = False
