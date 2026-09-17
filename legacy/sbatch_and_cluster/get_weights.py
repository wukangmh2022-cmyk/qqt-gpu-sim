import socket, os
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(15)
s.connect(('172.31.54.65', 29501))
buf = []
while True:
    chunk = s.recv(1048576)
    if not chunk: break
    buf.append(chunk)
os.makedirs('/root/private_data/qqt-gpu-sim/ckpt', exist_ok=True)
with open('/root/private_data/qqt-gpu-sim/ckpt/params_Pre-Train_Test.pkl', 'wb') as f:
    f.write(b''.join(buf))
print('DONE', os.path.getsize('/root/private_data/qqt-gpu-sim/ckpt/params_Pre-Train_Test.pkl'))
