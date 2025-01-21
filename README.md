# Introduction
This project implements a Tic-Tac-Toe game in Python with two modes: player vs. player and player vs. AI. The AI opponent uses a Random Forest machine learning model trained on a dataset of Tic-Tac-Toe moves.

##  Get Started
### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/yourusername/Tic-Tac-Toe-with-AI.git
```

### Install the required packages

 ```bash
 pip install -r requirements.txt
 ```

### Playing the game
1. **Navigate to the Notebooks directory:**

   ```bash
   cd Notebooks
   ```

2. **Launch Jupyter Notebook:**
   
  - Two-Player Mode
    
    ```bash
    jupyter AP_Tic_Tac_Toe.ipynb
    ```
  - AI Opponent Mode

    Ensure you have the trained model file (tic_tac_toe_rf_model.joblib) in the correct directory, then run:
    ```bash
    jupyter AP_Tictactoe_ml.ipynb
    ```

### Training the model
The random_forest_tictactoe_ml.py script demonstrates the process of training the Random Forest model used in the AI opponent mode. It includes data preprocessing, model training, and evaluation steps.

   ```bash
   jupyter random_forest_tictactoe_ml.ipynb
   ```

### Contributing
Contributions to improve the game or the AI model are welcome. Please feel free to submit pull requests or open issues for any bugs or enhancements.
