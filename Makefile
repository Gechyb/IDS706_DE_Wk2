# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# format code with black
format:
	black *.py

# Run linter (flake8 for Python files)
lint:
	flake8 --ignore=C,N data_analysis.py

# Run the data_analysis.py
run:
	python data_analysis.py
# Clean up Python cache files
clean:
	rm -rf __pycache__ .pytest_cache .coverage

# Default target
all: install format lint run