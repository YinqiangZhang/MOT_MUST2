import socket

# communicate with the matlab program using the socket
host = '127.0.0.1'
port = 65431
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
msg = 'client ok'

try:
    socket_tcp.connect((host, port))
    socket_tcp.sendall(bytes(msg, encoding='utf-8'))
except socket.error:
    print('fail to setup socket connection')
    socket_tcp.close()
