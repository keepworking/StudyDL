import numpy as np

def sigmoid(x):
    #뉴런에서 받은 데이터를 종합적인 신호로 변경하는 역할 계단함수나 시그모이드마냥
    #비선형 구조의 추력을 유지한다
    #0~1 사이의 범위가 룰인가보다.
    return 1/(1+np.exp(-x))

def identity_function(x):
    #출력측의 활성화 함수 하지만 아직 정의할 필요는 없다고한다.
    return x

def init_network():
    #신경망의 가중치나 구조를 미리 만들어둔다.
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2]) 
    
    return network

def forward(network,x):
    #각각의 행렬을 numpy로 계산한다.
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']
    # 
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)
