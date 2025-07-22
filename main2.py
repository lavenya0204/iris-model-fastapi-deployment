from fastapi import FastAPI

# Create a FastAPI application
app = FastAPI()

# Define a route at the root web address ("/")
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}





























# from fastapi import FastAPI,HTTPException
# from pydantic import BaseModel
# app = FastAPI()


# items =[]

# class Item(BaseModel):
#     text:str
#     is_done:bool = False

# @app.get("/")
# def root():
#     return{"hello":"world"}

# @app.get("/items", response_model=list[Item])
# def list_items(limit:int =10):
#     return items[0:limit]

# @app.post("/items")
# def create_items(item:Item):
#     items.append(item)
#     return items

# @app.get("/items/{item_id}", response_model=Item)
# def get_item(item_id:int)->Item:
#     if item_id<len(items):
#         item=items[item_id]
#         return item
#     else:
#         raise HTTPException(status_code=404, detail="item not found")