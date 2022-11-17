"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from SQLServerDemo import app
import flask
from flask import request
from flask import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from flask import Response
from sqlalchemy import Table,MetaData,Column  #結構對應類型
from sqlalchemy.dialects.mssql import NCHAR,NVARCHAR,VARCHAR #引用資料庫風格的欄位
from sqlalchemy.orm import mapper
from sqlalchemy.orm import sessionmaker
from flask_httpauth import HTTPBasicAuth
from flask_httpauth import HTTPTokenAuth

@app.route('/')
@app.route('/home')
@app.route('/index')
@app.route('/default')
def home():
    return render_template(
        'index.html',
        title='首頁',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
