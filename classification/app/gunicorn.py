import os

if os.environ.get('MODE') == 'dev':
    reload = True
# debug
debug = True
loglevel = 'debug'

bind = '0.0.0.0:8000'
workers = 2
timeout = 60

worker_class = 'uvicorn.workers.UvicornWorker'

# https://docs.gunicorn.org/en/stable/faq.html#how-do-i-avoid-gunicorn-excessively-blocking-in-os-fchmod
worker_tmp_dir = '/dev/shm'
