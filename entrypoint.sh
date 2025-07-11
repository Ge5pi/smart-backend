#!/bin/bash
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000 --timeout 300
