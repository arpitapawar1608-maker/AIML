from flask import Flask, render_template, request
import sqlite3
import requests
import json
import pickle
import numpy as np
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

DB = "db.sqlite"


# ---------------- DATABASE ----------------

def init_db():

    conn = sqlite3.connect(DB)

    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS emails(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT,
        input_text TEXT,
        detected_tone TEXT,
        generated_text TEXT,
        embedding BLOB,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()



# ---------------- OLLAMA ----------------

def ollama_generate(prompt):

    url="http://localhost:11434/api/generate"

    payload={
        "model":"llama3.2:latest",
        "prompt":prompt,
        "stream":False
    }

    r=requests.post(url,json=payload)

    return r.json()["response"]



def get_embedding(text):

    url="http://localhost:11434/api/embeddings"

    payload={
        "model":"nomic-embed-text:latest",
        "prompt":text
    }

    r=requests.post(url,json=payload)

    return np.array(
        r.json()["embedding"]
    )



# ---------------- CLASSIFICATION ----------------

def detect_tone(text):

    prompt=f"""

Classify this email request into one word:

formal
casual
apology
request


Text:
{text}

Return only the category.

"""

    result=ollama_generate(prompt)

    return result.strip().lower()



# ---------------- MEMORY ----------------


def save_email(
        email_type,
        input_text,
        tone,
        output
):

    emb=get_embedding(output)

    conn=sqlite3.connect(DB)

    c=conn.cursor()


    c.execute("""
    INSERT INTO emails
    VALUES(NULL,?,?,?,?,?,?)
    """,
    (
    email_type,
    input_text,
    tone,
    output,
    pickle.dumps(emb),
    datetime.now()
    ))


    conn.commit()
    conn.close()



# ---------------- RAG SEARCH ----------------


def retrieve_similar(query):

    conn=sqlite3.connect(DB)

    c=conn.cursor()


    c.execute("""
    SELECT generated_text,embedding
    FROM emails
    """)


    rows=c.fetchall()

    conn.close()


    if not rows:
        return []


    query_emb=get_embedding(query)


    scores=[]


    for text,blob in rows:

        emb=pickle.loads(blob)

        score=cosine_similarity(
            [query_emb],
            [emb]
        )[0][0]


        scores.append(
            (score,text)
        )


    scores.sort(reverse=True)


    return [
        x[1] for x in scores[:3]
    ]




# ---------------- ROUTES ----------------


@app.route("/",methods=["GET","POST"])
def index():

    result=""
    tone=""
    mode="generate"


    if request.method=="POST":


        mode=request.form["mode"]


        if mode=="generate":


            prompt=request.form["prompt"]


            tone=detect_tone(prompt)



            email_prompt=f"""

Write a complete email.

Tone: {tone}

User request:
{prompt}


Include subject and body.

"""


            result=ollama_generate(
                email_prompt
            )


            save_email(
                "generated",
                prompt,
                tone,
                result
            )



        else:


            incoming=request.form["incoming"]


            memories=retrieve_similar(
                incoming
            )


            context="\n\n".join(memories)



            reply_prompt=f"""

You are an email assistant.

Incoming email:

{incoming}


Previous writing style examples:

{context}


Write a reply matching the user's style.

"""


            result=ollama_generate(
                reply_prompt
            )


            tone="reply"


            save_email(
                "reply",
                incoming,
                tone,
                result
            )




    return render_template(
        "index.html",
        result=result,
        tone=tone,
        mode=mode
    )



if __name__=="__main__":

    init_db()

    app.run(
        debug=True
    )