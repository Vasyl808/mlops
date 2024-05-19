from marshmallow import Schema, fields, validate


class HeartDiseasePredictionInputSchema(Schema):
    age = fields.Integer(validate=validate.Range(min=29, max=77), required=True)
    sex = fields.Integer(validate=validate.OneOf([0, 1]), required=True)
    cp = fields.Integer(validate=validate.Range(min=0, max=3), required=True)
    trestbps = fields.Integer(validate=validate.Range(min=94, max=200), required=True)
    chol = fields.Integer(validate=validate.Range(min=126, max=564), required=True)
    fbs = fields.Integer(validate=validate.OneOf([0, 1]), required=True)
    thalach = fields.Integer(validate=validate.Range(min=71, max=202), required=True)
    restecg = fields.Integer(validate=validate.OneOf([0, 1, 2]), required=True)
    exang = fields.Integer(validate=validate.OneOf([0, 1]), required=True)
    oldpeak = fields.Float(validate=validate.Range(min=0, max=6.2), required=True)
    slope = fields.Integer(validate=validate.OneOf([0, 1, 2]), required=True)
    ca = fields.Integer(validate=validate.Range(min=0, max=4), required=True)
    thal = fields.Integer(validate=validate.OneOf([1, 2, 3]), required=True)
