from fastapi import FastAPI

# Create app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Example endpoint with parameter
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}

# Example "prediction" endpoint (dummy)
@app.get("/predict")
def predict(x: int, y: int):
    result = x + y
    return {"x": x, "y": y, "prediction": result}
