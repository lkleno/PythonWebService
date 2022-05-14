from flask import Flask
import joblib
import sklearn
app = Flask(__name__)

# Primer Web servica
@app.route("/<data>",methods=['GET'])
def GetPredictionRandomForest(data):
    model = joblib.load('RandomForestModel.pkl')
    data = data.strip('][').split(',') # V primeru da je podan string - '[1,2,3]' , drugace naredi brez tega.
    preparedData = list()
    for x in data:
        preparedData.append(int(x))
    return model.predict(preparedData)


if __name__ == "__main__":
    app.run()