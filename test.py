import unittest
from app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_predict_endpoint(self):
        data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "thalach": 150,
            "restecg": 0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
        response = self.client.post('/api/v1/predict', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('result', response.json)

    def test_invalid_json_data(self):
        data = "This is not JSON"
        response = self.client.post('/api/v1/predict', data=data, content_type='text/plain')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

    def test_invalid_data_schema(self):
        data = {
            "age": 63,
            "sex": "Male",  # Invalid type for 'sex'
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "thalach": 150,
            "restecg": 0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
        response = self.client.post('/api/v1/predict', json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)


if __name__ == '__main__':
    unittest.main()
