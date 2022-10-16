# Algorithm
2022.03~06

# 삽입정렬

~~~python
import random 
from timeit import default_timer as timer

def insertion_sort(A) :
  for i in range(1, len(A)) :
    loc = i - 1
    new_item = A[i]
    while loc >= 0 and new_item < A[loc] :
      A[loc + 1] = A[loc]
      loc -= 1
    A[loc + 1] = new_item
def test(A) :
  for i in range(1, len(A)) :
    if A[i-1] > A[i] :
      return False
  return True  

x = random.sample(range(10000),100)
start = timer()
insertion_sort(x)
print(timer()-start)
print(x)
print(test(x))
~~~

# 퀵정렬

~~~python
import random 
from timeit import default_timer as timer

def partition(A, p, r):
  x = A[r]
  i = p
  j = p
  
  if p < r:
    while j < r:
      if A[j] < x:
        A[i],A[j] = A[j],A[i]
        i = i + 1
      j = j + 1
    A[i],A[r] = A[r],A[i]

  return i
def qsort(A, p, r):
  if p < r:
    q = partition(A, p, r)
    qsort(A,p,q-1)
    qsort(A, q+1,r)

def quick_sort(A):
  qsort(A,0,len(A)-1)

def test(A):
  for i in range(1,len(A)):
    if A[i-1] > A[i]:
      return False
  return True

x = random.sample(range(10000),100)
start = timer()
quick_sort(x)
print(timer()-start)
print(x)
print(test(x))
~~~

# 이진 검색 트리 노드 삭제

~~~python
import random
from timeit import default_timer as timer

class Node(object):
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(node, key):
    if node is None: node = Node(key)
    elif key < node.key: node.left = insert(node.left, key)
    else: node.right = insert(node.right, key)
    return node
    
def search(node, key):
    if node is None or node.key == key: return node
    if key < node.key: return search(node.left, key)
    return search(node.right, key)

def delete(self, key):
    self.root, deleted = self._delete_value(self.root, key)
    return deleted

def _delete_value(self, node, key):
    if node is None:
        return node, False

    deleted = False
    # 해당 노드가 삭제할 노드일 경우
    if key == node.data:
        deleted = True
        # 삭제할 노드가 자식이 두개일 경우
        if node.left and node.right:
            # 오른쪽 서브 트리에서 가장 왼쪽에 있는 노드를 찾고 교체
            parent, child = node, node.right
            while child.left is not None:
                parent, child = child, child.left
            child.left = node.left
            if parent != node:
                parent.left = child.right
                child.right = node.right
            node = child
        # 자식 노드가 하나일 경우 해당 노드와 교체
        elif node.left or node.right:
            node = node.left or node.right
        # 자식 노드가 없을 경우 그냥 삭제
        else:
            node = None
    elif key < node.data:
        node.left, deleted = self._delete_value(node.left, key)
    else:
        node.right, deleted = self._delete_value(node.right, key)
    return node, deleted
      
x = random.sample(range(5000), 1000)
value = x[800]

root = None
for i in x:
    root = insert(root, i)

start = timer()
found = search(root, value)
print(timer() - start)

if found is not None:
    print('value', value, 'found', found.key)
    print(True if found.key == value else False)
~~~

# 피보나치 수열

~~~python
import random

def FIB(n):
  f=[1,1]
  if n==1 or n==2:
    return 1
  else:
    for i in range(2,n):
      a = f[i-1] + f[i-2]
      f.append(a)
    return f

x = random.randrange(1,11)
print(x,FIB(x))
~~~

# 행렬 경로

~~~python
import random

def MP(n):

  m = [[0 for j in range(n)] for i in range(n)]
  for i in range(1,n):
    for j in range(1,n):
      m[i][j] = random.randrange(1,20)

  c = [[0 for j in range(n)] for i in range(n)]
  for i in range(0,n):
    c[i][0] = 0
  for j in range(0,n):
    c[0][j] = 0
  for i in range(1,n):
    for j in range(1,n):
      c[i][j] = m[i][j] + max(c[i-1][j],c[i][j-1])

  for i in m:
    print(i,end="\n")
  print(c[n-1][n-1])

n = random.randrange(5,7)
MP(n)
~~~

# 위상정렬 2

~~~python

class Graph:

    def __init__(self):
        self.edges = {}

    def addNode(self, node):
        self.edges[node] = []

    def addEdge(self, node1, node2):
        self.edges[node1] += [node2]

    def getSub(self, node):
        return self.edges[node]

    def DFSrecu(self, start, path):

        for node in self.getSub(start):
            if node not in path:
                path = self.DFSrecu(node, path)

        if start not in path:
            path += [start]

        return path

    def topological_sort(self, start):
        topo_ordering_list = self.DFSrecu(start, [])
        for node in g.edges:
            if node not in topo_ordering_list:
                topo_ordering_list = g.DFSrecu(node, topo_ordering_list)
        return topo_ordering_list


if __name__ == "__main__":
    g = Graph()
    for node in ['냄비에 물 붓기', '점화', '라면 넣기', '라면 봉지 뜯기', '수프 넣기', '계란 넣기']:
        g.addNode(node)

    g.addEdge("냄비에 물 붓기", "점화")
    g.addEdge("라면 봉지 뜯기", "라면 넣기")
    g.addEdge("점화", "라면 넣기")
    g.addEdge("점화", "수프 넣기")
    g.addEdge("점화", "계란 넣기")
    g.addEdge("라면 넣기", "계란 넣기")
    g.addEdge("수프 넣기", "계란 넣기")

    last_path1 = g.topological_sort("점화")
    last_path1.reverse()
    last_path2 = g.topological_sort("수프 넣기")
    last_path2.reverse()

    print("Start From 점화: ",last_path1)
    print("start From 수프 넣기: ",last_path2)
~~~

# 좌표지도
![image](https://user-images.githubusercontent.com/114469334/196025890-6a4dd359-2ce6-4647-8211-f69da94e9bfe.png)

# A*알고리즘 heuristic

~~~python
import heapq
import math

class Queue(object):
  def __init__(self):
    self.elements = []

  def length(self):
    return len(self.elements)
  
  def push(self, x, priority):                                                  # heap 원소 추가 (좌표 , 순번)/ 0(logN) 시간복잡도
    heapq.heappush(self.elements, (priority, x))

  def pop(self):                                                                # heap 원소 삭제 / 0(logN) 시간복잡도
    return heapq.heappop(self.elements)[1]

grid = [[0,0,0,0,0,0,0,0,0,0,],                                                 #0은 벽
        [0,1,1,1,1,1,1,1,1,0,],
        [0,3,0,0,0,0,2,0,1,0,],
        [0,3,1,1,1,0,1,0,1,0,],
        [0,0,1,0,1,0,1,1,1,0,],
        [0,1,1,0,1,1,1,0,1,0,],
        [0,0,0,0,0,0,0,0,0,0,],]

start = (1,5)                                                                   # 출발점 
goal = (8,1)                                                                    # 목표점

queue = Queue()
queue.push(start, 0)  
came_from = {}
cost_so_far = {}
cost_so_far[start] = 0

def calc_cost(current, next):                                                   # 코스트 계산
  (x,y) = next
  return cost_so_far[current] + grid[y][x]

def heuristic(current, next):                                                   # 휴리스틱(h) _ 대각선 길이 출력
  (x1,y1) = goal
  (x2,y2) = next
  dx = x1 - x2
  dy = y1 - y2
  return math.sqrt(dx*dx + dy*dy)                                               


while queue.length() > 0:
  current = queue.pop()                                                         # 큐에서 제거

  if current == goal:                                                           # 목표에 도달하면 종료
    break

  (x,y) = current
  candidates = [(x+1,y), (x,y-1), (x-1,y), (x,y+1)] # 이동 후보들
  for next in [(h, v) for h, v in candidates if grid[v][h] != 0]:
    new_cost = calc_cost(current, next)                                         # 코스트 재계산 
    if next not in came_from or new_cost < cost_so_far[next]:                   # 방문한적 없음 or 계산비용 < 누적비용
      print(heuristic(goal, next))                                              # 유클리디언 거리 출력
      queue.push(next, new_cost + heuristic(goal, next))                        # heapq에 추가
      cost_so_far[next] = new_cost                                              # 계산비용 업데이트
      came_from[next] = current                                                 # 현재 위치를 지나온곳으로 업데이트 

current = goal                                                                  # 경로 goal에서 시작
path = []
while current is not start:                                                     # 경로 역추적 - start가 될때까지 반복
  path.append(current)                                                          # path에 추가
  current = came_from[current]                                  
path.reverse()                                                                  # 경로 반전
print(path)                                                                     # start에서 goal까지의 경로
~~~

# A*알고리즘 manhattan

~~~python
import heapq

class Queue(object):
  def __init__(self):
    self.elements = []

  def length(self):
    return len(self.elements)
  
  def push(self, x, priority):                                                  # heap 원소 추가 (좌표 , 순번)/ 0(logN) 시간복잡도
    heapq.heappush(self.elements, (priority, x))

  def pop(self):                                                                # heap 원소 삭제 / 0(logN) 시간복잡도
    return heapq.heappop(self.elements)[1]

grid = [[0,0,0,0,0,0,0,0,0,0,],                                                 #0은 벽
        [0,1,1,1,1,1,1,1,1,0,],
        [0,3,0,0,0,0,2,0,1,0,],
        [0,3,1,1,1,0,1,0,1,0,],
        [0,0,1,0,1,0,1,1,1,0,],
        [0,1,1,0,1,1,1,0,1,0,],
        [0,0,0,0,0,0,0,0,0,0,],]

start = (1,5)                                                                   # 출발점 
goal = (8,1)                                                                    # 목표점

queue = Queue()
queue.push(start, 0)  
came_from = {}
cost_so_far = {}
cost_so_far[start] = 0

def calc_cost(current, next):                                                   # 코스트 계산
  (x,y) = next
  return cost_so_far[current] + grid[y][x]

def manhattan(current, next):                                                   # 맨하탄 거리 _ x y 거리의 절대값의 합
  (x1,y1) = goal
  (x2,y2) = next
  dx = x1 - x2
  dy = y1 - y2
  return abs(dx) + abs(dy)                                            


while queue.length() > 0:
  current = queue.pop()                                                         # 큐에서 제거

  if current == goal:                                                           # 목표에 도달하면 종료
    break

  (x,y) = current
  candidates = [(x+1,y), (x,y-1), (x-1,y), (x,y+1)] # 이동 후보들
  for next in [(h, v) for h, v in candidates if grid[v][h] != 0]:
    new_cost = calc_cost(current, next)                                         # 코스트 재계산 
    if next not in came_from or new_cost < cost_so_far[next]:                   # 방문한적 없음 or 계산비용 < 누적비용
      print(manhattan(goal, next))                                              # 맨하탄 거리 출력
      queue.push(next, new_cost + manhattan(goal, next))                        # heapq에 추가
      cost_so_far[next] = new_cost                                              # 계산비용 업데이트
      came_from[next] = current                                                 # 현재 위치를 지나온곳으로 업데이트 

current = goal                                                                  # 경로 goal에서 시작
path = []
while current is not start:                                                     # 경로 역추적 - start가 될때까지 반복
  path.append(current)                                                          # path에 추가
  current = came_from[current]                                  
path.reverse()                                                                  # 경로 반전
print(path)                                                                     # start에서 goal까지의 경로
~~~

# 쉘 정렬

~~~python
from random import randint

def shell_sort_one(arr,first, last, h):
  i = first + h                                 # h 수치만큼 차이 발생
  while i <= last:                              # 행수치만큼 반복
    val = arr[i]                                # 배열 저장
    pos = i                                     # 위치 저장 (h차이를 내고 저장하기 위해)
    while pos > first and arr[pos - h] > val:   
      arr[pos] = arr[pos-h]                     
      pos -= h
    arr[pos] = val
    i += h

def shell_sort_two(arr):
  n = len(arr)                                  # 행 크기 지정
  h_list = [57,23,10,4,1]                       # 지정된 h 저장
  for j in range(len(h_list)):
    h = h_list[j]                               # h 변환
    for i in range(0,h):
      shell_sort_one(arr,i,n-1,h)               # 정렬
  return arr

lst = [randint(1,100) for i in range(100)]      # 1~100까지 랜덤한 정수 100개 생성
print(shell_sort_two(lst))                      # 결과물 출력
~~~


# 수업을 이수하면서
A* 알고리즘과 유클리디언, 맨하탄 거리 등의 알고리즘끼리 비교하는 부분이 가장 기억에 남는다. 경로찾기 부분이 가장 직관적으로 다가왔기에 그런것 같다.
또한 수업을 이수하면서 python 코딩방식에 어느정도 익숙해진 시간이 되었다.
