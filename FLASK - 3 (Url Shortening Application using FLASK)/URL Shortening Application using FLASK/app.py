
######################################################
from flask import Flask,request,render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pyshorteners
import os
shorter=""
url=""

#######################################################
app=Flask(__name__)
#######################################################

basedir=os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///'+os.path.join(basedir,'data.sqllite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db=SQLAlchemy(app)  
Migrate(app,db)

class Data(db.Model):
    __tablename__='urlshortener'
    id=db.Column(db.Integer,primary_key=True)
    url=db.Column(db.String(100))
    shorter = db.Column(db.String(100))
    
    def __init__(self,url,shorter):
        self.url=url
        self.shorter=shorter
        
@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/',methods=["GET","POST"])
def home():
    global shorter,url
    if request.method=='POST':
        url=request.form.get('name')
        s=pyshorteners.Shortener()
        shorter=s.tinyurl.short(url)
        val_pass=Data(url,shorter)
        db.session.add(val_pass)
        db.session.commit()
        
    return render_template('Home Page.html',n=shorter)

@app.route('/Earlier Details')
def Earlier_Details():
    allurls=Data.query.all()
    return render_template('Earlier Details.html',allurls=allurls)

###########################################################

if __name__=="__main__":
    app.run(debug=True)
