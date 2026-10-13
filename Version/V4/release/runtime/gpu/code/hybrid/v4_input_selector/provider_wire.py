"""Bounded local provider RPC; no network listeners or model imports."""
import socket
import time

from .worker_contract import recv_message, send_message


def send(sock, value, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Provider send deadline')
    sock.settimeout(remaining)
    send_message(sock, value)


class Client:
    def __init__(self, sock):
        self.sock, self.sequence = sock, 0

    def call(self, operation, arguments, deadline):
        self.sequence += 1
        send(self.sock, dict(seq=self.sequence, operation=operation, arguments=arguments, deadline=deadline), deadline)
        response = recv_message(self.sock, deadline=deadline)
        if not isinstance(response, dict) or response.get('seq') != self.sequence or type(response.get('ok')) is not bool:
            raise ValueError('Provider response identity differs')
        if not response['ok']:
            raise RuntimeError('Provider operation failed: ' + str(response.get('error')))
        return response['result']


def reply(sock, request, *, result=None, error=None):
    send(sock, dict(seq=request['seq'], ok=error is None, result=result, error=error), request['deadline'])


def validate_request(request, previous):
    import math
    if not isinstance(request, dict) or set(request) != {'seq', 'operation', 'arguments', 'deadline'}:
        raise ValueError('Malformed provider request')
    if type(request['seq']) is not int or request['seq'] != previous + 1 or not isinstance(request['arguments'], dict):
        raise ValueError('Provider request sequence/arguments differ')
    deadline = request['deadline']
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
        raise TimeoutError('Expired or invalid provider request')
    return request['seq']
