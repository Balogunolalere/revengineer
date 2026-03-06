from instagrapi.types import User
u = User(pk="123", username="test", full_name="Test")
print("Has model_dump:", hasattr(u, "model_dump"))
print("Has dict:", hasattr(u, "dict"))
