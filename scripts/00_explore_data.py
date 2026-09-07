from sklearn.datasets import load_breast_cancer

d = load_breast_cancer(as_frame=True)

print("Shape:", d.frame.shape)
print("Target names:", d.target_names)
print("\nClass proportion:")
print(d.frame["target"].value_counts(normalize=True))