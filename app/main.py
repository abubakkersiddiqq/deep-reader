from fastapi import FastAPI

app = FastAPI()


@app.get('/')
def health_status():
    return {'status': 'ok'}