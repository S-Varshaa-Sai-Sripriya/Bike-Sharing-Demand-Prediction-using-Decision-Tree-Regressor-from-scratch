from setuptools import setup, find_packages

setup(
    name="bike_sharing_dtreg",
    version="0.1.0",
    description="Decision Tree Regressor project for predicting bike sharing demand.",
    author="S Varshaa Sai Sripriya",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "train_model=src.training.train:main",
            "evaluate_model=src.evaluation.evaluate:main",
        ],
    },
    python_requires=">=3.10",
)
