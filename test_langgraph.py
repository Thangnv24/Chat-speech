from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from pprint import pprint

# 1. Định nghĩa trạng thái (State)
class AgentState(TypedDict):
    """Trạng thái cho đồ thị."""
    query: str
    result: str
    # Sử dụng operator.add để cập nhật số lần thử
    num_tries: Annotated[int, operator.add] 

# 2. Định nghĩa Node (Thực hiện công việc)
def generate_response(state: AgentState) -> AgentState:
    """Tạo phản hồi và cập nhật trạng thái."""
    print(f"\n[GENERATOR] Đang chạy lần thứ: {state['num_tries'] + 1}")
    
    if state["num_tries"] == 0:
        # Lần chạy đầu tiên: Cố ý trả về kết quả chưa hoàn chỉnh
        response = "Đây là kết quả chưa hoàn chỉnh."
        tries_increment = 1
    else:
        # Lần chạy sau: Giả lập kết quả đã sửa
        response = "Đây là kết quả sau khi đã sửa chữa và hoàn chỉnh."
        tries_increment = 0 # Không cần tăng nữa nếu đã hoàn thành

    # Node luôn phải trả về Dict để cập nhật State
    return {"result": response, "num_tries": tries_increment}

# 3. Định nghĩa Hàm Routing (Quyết định luồng)
def route_check(state: AgentState) -> str:
    """
    Hàm này chỉ dùng cho conditional edge.
    Nó đọc trạng thái và quyết định bước tiếp theo.
    """
    if "chưa hoàn chỉnh" in state["result"]:
        print("-> [CHECKER] Đã phát hiện kết quả chưa hoàn chỉnh. CHUYỂN ĐẾN TÁI TẠO (re_run).")
        return "re_run" 
    else:
        print("-> [CHECKER] Kết quả đã hoàn chỉnh. KẾT THÚC (end).")
        return "end" 

# 4. Xây dựng Đồ thị Trạng thái
workflow = StateGraph(AgentState)

# Thêm Node (chỉ có 1 node thực hiện công việc)
workflow.add_node("generator", generate_response)

# Thiết lập điểm bắt đầu
workflow.set_entry_point("generator")

# Thêm Conditional Edge (Rẽ nhánh)
# Quyết định rẽ nhánh được thực hiện ngay sau khi 'generator' chạy
workflow.add_conditional_edges(
    "generator",      # Node nguồn (Source node)
    route_check,      # Hàm quyết định (Decision function)
    {
        "re_run": "generator", # Nếu 're_run', quay lại 'generator' (Vòng lặp)
        "end": END             # Nếu 'end', kết thúc luồng
    }
)

# 5. Biên dịch (Compile) và chạy
app = workflow.compile()

print("--- Kết quả LangGraph đã sửa lỗi ---")
initial_state = {"query": "Vấn đề cần xử lý", "result": "", "num_tries": 0}

# Chạy và lấy tất cả các bước (để thấy vòng lặp)
# Sử dụng stream để xem từng bước được thực hiện.
for step in app.stream(initial_state):
    pprint(step)

final_state = app.invoke(initial_state)
print("\n--- Trạng thái cuối cùng của ứng dụng ---")
print(f"Trạng thái cuối cùng: {final_state['result']}")