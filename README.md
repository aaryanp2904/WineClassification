
# Setting up a Virtual Environment and Installing Dependencies

Follow these steps to set up a virtual environment (`venv`) and install the required dependencies from the `requirements.txt` file.

## 1. Create a Virtual Environment

Navigate to your project directory and create a new virtual environment using the following command:

### On Windows:

```bash
python -m venv venv
```

### On macOS/Linux:

```bash
python3 -m venv venv
```

This will create a `venv` directory in your project folder containing the virtual environment.

## 2. Activate the Virtual Environment

Once the virtual environment is created, activate it.

### On Windows:

```bash
.\venv\Scripts\activate
```

### On macOS/Linux:

```bash
source venv/bin/activate
```

After activation, you should see the environment's name (`venv`) in your terminal prompt.

## 3. Install Dependencies from `requirements.txt`

With the virtual environment activated, install the required dependencies listed in the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

This will install all the packages specified in `requirements.txt` into your virtual environment.

## 4. Verify Installation

To verify that the packages were installed correctly, you can list the installed packages with:

```bash
pip list
```

This will show you all the packages installed in your virtual environment.

## 5. Deactivate the Virtual Environment

Once you're done working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```

This will return you to your global Python environment.

---

By following these steps, you'll have a fully set-up virtual environment with all the required dependencies installed from `requirements.txt`.


# Structure

The notebook contains some initial data exploration.

The main.py contains an Ensemble / Majority Voting model using a neural net, random forest and LLM.