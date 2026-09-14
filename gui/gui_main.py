"""gui_main.py — ADOFAI 谱面生成器 · 原生 Mica 窗口壳 + WebView2 网页界面

两层结构（原生外壳 + 本地 HTTP 服务）：
  - 第 1 层：Win32 无边框窗口 + DWM Mica 背景（系统级毛玻璃，透桌面壁纸）
  - 第 2 层：窗口里嵌 WebView2（系统自带的 Chromium 内核，零额外体积）
           加载 gui/index.html（HTML/CSS 画的 DSH 风格界面）

网页 ↔ 宿主 通过 WebView2 WebMessage 通信：
  网页 -> 宿主：{type:'generate'|'drag'|'close', ...}
  宿主 -> 网页：{type:'log'|'done', ...}

双击请用「启动.bat」（pythonw 无黑框）。本文件也可直接 pythonw 运行。
"""
from __future__ import annotations
import os, sys, json, subprocess, threading, traceback, collections, time
import ctypes
import ctypes.wintypes as wt

# ---------- 路径（必须在 import comtypes 之前把 venv 加进 sys.path）----------
ROOT = os.path.dirname(os.path.abspath(__file__))          # .../gui
# PROJ 默认 = gui 的上级目录（开发态直接 pythonw 跑 gui_main.py 时就是项目根）。
# 单文件 exe 把 gui 释放到 Temp 后，用 ADOFAI_PROJ_ROOT 指向真实项目根，
# 让 web_server 仍从磁盘 app/ 导入、PORTABLE_ROOT 正确指向 venv/data。
PROJ = os.environ.get("ADOFAI_PROJ_ROOT") or os.path.dirname(ROOT)
VENV_SP = os.path.join(PROJ, "venv", "Lib", "site-packages")
if VENV_SP not in sys.path:
    sys.path.insert(0, VENV_SP)
# 运行时数据目录：装到 C:\Program Files 后普通用户无写权限，运行时产生的缓存/预览/日志
# 重定向到 %LOCALAPPDATA%\ADOFAI_Diffusion（卸载时一并清掉）。
RUNTIME_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ADOFAI_Diffusion")
# 让推理子进程也能找到数据目录（不依赖外部 bat 设置）
os.environ.setdefault("ADOFAI_DATA_DIR", RUNTIME_DIR)

# 复用网页后端的生成/分离/训练逻辑（同一份代码，原生 GUI 直接 import 调用）
APP_DIR = os.path.join(PROJ, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
# 直接 import 整个 web_server 模块：分离(separate_all/preview_track)、训练(_start_training/
# _start_vfx_training)、训练状态机(TRAIN_STATE/VFX_STATE) 与预览目录(PREVIEW_DIR) 都能复用
import web_server as ws

# 🔴 Python 3.13 venv 的 pyvenv.cfg 用相对路径找 base python（如 ..\..\python313\python.exe），
#    但 3.13 把相对路径基于 CWD 解析而非 venv 自身位置。GUI 从 gui/ 启动时 CWD 是 gui/，
#    导致 venv 子进程找不到 python313 → "did not find executable"。
#    修复：启动即切 CWD 到项目根目录（ffmpeg/data/checkpoints/venv 都在这）。
os.chdir(PROJ)

# ---------- 调试日志（真机诊断用，双击黑屏时看这个文件）----------
DEBUG_LOG = os.path.join(ROOT, "gui_debug.log")
def dbg(msg):
    try:
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (ts, msg))
    except Exception:
        pass

# ---------- 提前把线程初始化为 STA（WebView2 硬性要求单线程套间）----------
ole32 = ctypes.windll.ole32
ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
ole32.CoInitializeEx.restype = ctypes.HRESULT
try:
    _hr = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
    dbg("CoInitializeEx(STA) hr=%s" % hex(_hr & 0xFFFFFFFF))
except Exception as e:
    dbg("CoInitializeEx 异常: %r" % (e,))

import comtypes
import comtypes.client

# ---------- 其余路径 ----------
TLB  = os.path.join(ROOT, "WebView2.tlb")
LOADER = os.path.join(ROOT, "WebView2Loader.dll")
HTML_PATH = os.path.join(ROOT, "index.html")
PY313 = os.path.join(PROJ, "python313", "python.exe")
DATA_DIR = RUNTIME_DIR
WEBVIEW_CACHE = os.path.join(DATA_DIR, "webview_cache")
INFERENCE = os.path.join(PROJ, "app", "training", "inference_stage2.py")
TRAIN_SHAPE_SCRIPT = os.path.join(APP_DIR, "training", "train_shape.py")
SHAPE_LOG = os.path.join(DATA_DIR, "shape_train.log")
SHAPE_STATE = {"proc": None, "status": "idle", "started": 0.0}

# ---------- comtypes 生成 WebView2 接口 ----------
WV = comtypes.client.GetModule(TLB)

# ---------- ctypes 类型别名（ctypes.wintypes 缺这些）----------
LRESULT = ctypes.c_ssize_t
HCURSOR = wt.HANDLE
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

# ---------- Win32 常量 ----------
WM_DESTROY = 0x0002
WM_SIZE = 0x0005
WM_CLOSE = 0x0010
WM_SETICON = 0x0080
ICON_BIG = 1
ICON_SMALL = 0
WM_ERASEBKGND = 0x0014
WM_NCLBUTTONDOWN = 0x00A1
HTCAPTION = 2
WM_APP_LOG = 0x8000 + 1
WM_APP_DONE = 0x8000 + 2
WM_APP_DRAG = 0x8000 + 3
WM_APP_PICK = 0x8000 + 4
WM_APP_RESIZE = 0x8000 + 5
WM_CONTEXTMENU = 0x007B   # 彻底屏蔽窗口右键菜单

# 窗口缩放 / 最小最大化
WM_NCCALCSIZE    = 0x0083
WM_NCHITTEST     = 0x0084
WM_GETMINMAXINFO = 0x0024
WM_SETCURSOR     = 0x0020
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE  = 9
HTCLIENT = 1
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17
IDC_SIZEWE   = 32644
IDC_SIZENS   = 32645
IDC_SIZENWSE = 32642
IDC_SIZENESW = 32643

# HTML 边缘热区 → 非客户区命中类型（发起标准 Windows 缩放）
EDGE_HT = {
    "left": HTLEFT, "right": HTRIGHT, "top": HTTOP,
    "bottom": HTBOTTOM, "topleft": HTTOPLEFT, "topright": HTTOPRIGHT,
    "bottomleft": HTBOTTOMLEFT, "bottomright": HTBOTTOMRIGHT,
}
WS_THICKFRAME  = 0x00040000
WS_SYSMENU     = 0x00080000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WIN_MIN_W, WIN_MIN_H = 560, 420
RESIZE_BORDER = 6                  # WebView2 内缩像素数（留出缩放命中区，须 >= 命中区 b）

DWMWA_SYSTEMBACKDROP_PREFERENCE = 38
DWMSBT_MAINWINDOW = 2          # Mica
DWMSBT_TRANSIENTWINDOW = 3     # Acrylic
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2
DWMWA_USE_IMMERSIVE_DARK_MODE = 20

CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
CW_USEDEFAULT = 0x80000000

WND_CLASS = "ADOFAI_GUI_Class"
WIN_W, WIN_H = 920, 600

# ---------- 全局状态 ----------
HWND_MAIN = None
WEBVIEW_ENV = None
WEBVIEW_CTRL = None
WEBVIEW = None
WEBVIEW_MSG_HANDLER = None
ALIVE = []                     # 保活 COM handler，防 GC
log_q = collections.deque()
log_lock = threading.Lock()
gen_thread = None
last_out = None
last_code = None
last_error = None
pick_ctx = {"dest": ""}
last_zoomed = False

# ---------- WNDCLASSEX 结构 ----------
class WNDCLASSEX(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("style", wt.UINT),
                ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE),
                ("hIcon", wt.HICON), ("hCursor", HCURSOR),
                ("hbrBackground", wt.HBRUSH), ("lpszMenuName", wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON)]

# ---------- 加载 DLL / 设置 argtypes ----------
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
dwmapi = ctypes.windll.dwmapi
ole32 = ctypes.windll.ole32

user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEX)]
user32.RegisterClassExW.restype = wt.ATOM

# 图标加载
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x10
LR_DEFAULTSIZE = 0x40

user32.LoadImageW.argtypes = [wt.HINSTANCE, wt.LPCWSTR, wt.UINT, ctypes.c_int, ctypes.c_int, wt.UINT]
user32.LoadImageW.restype = wt.HANDLE

user32.SendMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.SendMessageW.restype = LRESULT

user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.CreateWindowExW.restype = wt.HWND

user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
user32.GetMessageW.restype = ctypes.c_long
user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
user32.TranslateMessage.restype = wt.BOOL
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.DestroyWindow.argtypes = [wt.HWND]
user32.DestroyWindow.restype = wt.BOOL
user32.PostQuitMessage.argtypes = [ctypes.c_int]
user32.PostQuitMessage.restype = None
user32.LoadCursorW.argtypes = [wt.HINSTANCE, wt.LPCWSTR]
user32.LoadCursorW.restype = HCURSOR
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.ShowWindow.restype = wt.BOOL
user32.UpdateWindow.argtypes = [wt.HWND]
user32.UpdateWindow.restype = wt.BOOL
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.PostMessageW.restype = wt.BOOL
user32.ReleaseCapture.argtypes = []
user32.ReleaseCapture.restype = wt.BOOL
user32.SendMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.SendMessageW.restype = LRESULT
user32.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
user32.GetCursorPos.restype = wt.BOOL
user32.IsZoomed.argtypes = [wt.HWND]
user32.IsZoomed.restype = wt.BOOL
user32.SetCursor.argtypes = [HCURSOR]
user32.SetCursor.restype = HCURSOR
user32.LOWORD = lambda v: v & 0xFFFF
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowRect.restype = wt.BOOL

dwmapi.DwmSetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long

comdlg32 = ctypes.windll.comdlg32

class MINMAXINFO(ctypes.Structure):
    _fields_ = [("ptReserved", wt.POINT), ("ptMaxSize", wt.POINT),
                ("ptMaxPosition", wt.POINT), ("ptMinTrackSize", wt.POINT),
                ("ptMaxTrackSize", wt.POINT)]

class MARGINS(ctypes.Structure):
    _fields_ = [("cxLeftWidth", ctypes.c_int), ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int), ("cyBottomHeight", ctypes.c_int)]

dwmapi.DwmExtendFrameIntoClientArea.argtypes = [wt.HWND, ctypes.POINTER(MARGINS)]
dwmapi.DwmExtendFrameIntoClientArea.restype = ctypes.c_long

class OPENFILENAMEW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wt.DWORD), ("hwndOwner", wt.HWND), ("hInstance", wt.HINSTANCE),
        ("lpstrFilter", wt.LPCWSTR), ("lpstrCustomFilter", wt.LPCWSTR),
        ("nMaxCustFilter", wt.DWORD), ("nFilterIndex", wt.DWORD),
        ("lpstrFile", wt.LPWSTR), ("nMaxFile", wt.DWORD),
        ("lpstrFileTitle", wt.LPWSTR), ("nMaxFileTitle", wt.DWORD),
        ("lpstrInitialDir", wt.LPCWSTR), ("lpstrTitle", wt.LPCWSTR),
        ("Flags", wt.DWORD), ("nFileOffset", wt.WORD), ("nFileExtension", wt.WORD),
        ("lpstrDefExt", wt.LPCWSTR), ("lCustData", ctypes.POINTER(wt.LPARAM)),
        ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wt.LPCWSTR),
        ("pvReserved", ctypes.c_void_p), ("dwReserved", wt.DWORD), ("FlagsEx", wt.DWORD)]

comdlg32.GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
comdlg32.GetOpenFileNameW.restype = wt.BOOL
OFN_FILEMUSTEXIST = 0x1000
OFN_PATHMUSTEXIST = 0x0800
OFN_NOCHANGEDIR = 0x0008

AUDIO_FILTER = "音频文件\0*.mp3;*.wav;*.ogg;*.flac;*.m4a;*.opus\0所有文件\0*.*\0\0"


def native_open_file():
    """原生打开文件对话框，返回选中文件的绝对路径或 None。"""
    buf = ctypes.create_unicode_buffer(2048)
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.hwndOwner = HWND_MAIN
    ofn.lpstrFilter = wt.LPCWSTR(AUDIO_FILTER)
    ofn.lpstrFile = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile = 2048
    ofn.lpstrTitle = wt.LPCWSTR("选择音频文件")
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
    if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return buf.value
    return None

kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
kernel32.GetModuleHandleW.restype = wt.HINSTANCE


# ---------- DWM 辅助 ----------
def set_backdrop(hwnd, kind):
    v = ctypes.c_int(kind)
    dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_SYSTEMBACKDROP_PREFERENCE,
                                ctypes.byref(v), ctypes.sizeof(v))

def set_round(hwnd):
    v = ctypes.c_int(DWMWCP_ROUND)
    dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
                                ctypes.byref(v), ctypes.sizeof(v))

def set_dark(hwnd):
    v = ctypes.c_int(1)
    dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                                ctypes.byref(v), ctypes.sizeof(v))

def extend_frame(hwnd):
    """把 DWM 玻璃/Acrylic 材质延伸到整个客户区（否则 WS_THICKFRAME 下 WebView2 透明背景糊成白底）。"""
    m = MARGINS(-1, -1, -1, -1)
    r = dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(m))
    dbg("DwmExtendFrameIntoClientArea hr=%s" % hex(r & 0xFFFFFFFF))


# ---------- WebView2 引导（C 导出 + comtypes 接口）----------
_loader = ctypes.windll.LoadLibrary(LOADER)
_create_env = _loader.CreateCoreWebView2EnvironmentWithOptions
_create_env.argtypes = [wt.LPCWSTR, wt.LPCWSTR, ctypes.c_void_p,
                        ctypes.POINTER(WV.ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler)]
_create_env.restype = ctypes.HRESULT


class EnvCompleted(comtypes.COMObject):
    _com_interfaces_ = [WV.ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler]
    def Invoke(self, result, environment):
        global WEBVIEW_ENV
        rh = hex(result & 0xFFFFFFFF) if isinstance(result, int) else result
        dbg("EnvCompleted.Invoke result=%s env=%s" % (rh, "OK" if environment else "None"))
        if result < 0 or environment is None:
            dbg("环境创建失败 -> 弹窗提示")
            ctypes.windll.user32.MessageBoxW(None,
                u"WebView2 环境创建失败\nHRESULT=%s\n\n可能原因：本机 WebView2 Runtime 与内置 SDK 不兼容，或首次运行需要联网下载。\n详细请查看 gui 目录下的 gui_debug.log" % rh,
                u"ADOFAI GUI", 0x10)
            return
        WEBVIEW_ENV = environment
        ALIVE.append(self)
        h = ControllerCompleted()
        ALIVE.append(h)
        iface = h.QueryInterface(WV.ICoreWebView2CreateCoreWebView2ControllerCompletedHandler)
        dbg("CreateCoreWebView2Controller hwnd=%s" % HWND_MAIN)
        environment.CreateCoreWebView2Controller(int(HWND_MAIN), iface)


class ControllerCompleted(comtypes.COMObject):
    _com_interfaces_ = [WV.ICoreWebView2CreateCoreWebView2ControllerCompletedHandler]
    def Invoke(self, result, controller):
        global WEBVIEW_CTRL, WEBVIEW, WEBVIEW_MSG_HANDLER
        rh = hex(result & 0xFFFFFFFF) if isinstance(result, int) else result
        dbg("ControllerCompleted.Invoke result=%s ctrl=%s" % (rh, "OK" if controller else "None"))
        if result < 0 or controller is None:
            dbg("控制器创建失败 -> 弹窗提示")
            ctypes.windll.user32.MessageBoxW(None,
                u"WebView2 控制器创建失败\nHRESULT=%s\n\n可能父窗口句柄无效，或 WebView2 Runtime 异常。\n详见 gui_debug.log" % rh,
                u"ADOFAI GUI", 0x10)
            return
        WEBVIEW_CTRL = controller
        ALIVE.append(self)
        try:
            # 透明背景：让 Mica 透出来（DSH 侧栏透壁纸模糊的关键）
            ctrl2 = controller.QueryInterface(WV.ICoreWebView2Controller2)
            if os.environ.get("ADOFAI_GUI_OPAQUE"):
                # 诊断模式：不透明深色背景，规避 WebView2 透明渲染 bug
                ctrl2.DefaultBackgroundColor = WV.COREWEBVIEW2_COLOR(A=255, R=20, G=20, B=24)
                dbg("不透明调试背景已设 (A=255)")
            else:
                ctrl2.DefaultBackgroundColor = WV.COREWEBVIEW2_COLOR(A=0, R=0, G=0, B=0)
                dbg("透明背景已设 (A=0)")
        except Exception as e:
            dbg("设背景色失败(非致命): %r" % (e,))
        try:
            controller.Bounds = wt.RECT(0, 0, WIN_W, WIN_H)
            webview = controller.CoreWebView2
            WEBVIEW = webview
            dbg("已获取 CoreWebView2")
            # 虚拟主机映射：让 HTML 内的 https://gui.assets/ 指向 gui/ 文件夹
            try:
                _access_kind = getattr(WV, 'COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW', 2)
                webview.SetVirtualHostNameToFolderMapping(
                    "gui.assets", ROOT, _access_kind)
                dbg("虚拟主机映射已设: gui.assets -> %s" % ROOT)
            except Exception as e:
                dbg("虚拟主机映射失败(非致命): %r" % (e,))
            settings = webview.Settings
            settings.IsWebMessageEnabled = True
            try:
                settings.IsContextMenuEnabled = False  # 禁用 WebView2 右键菜单
            except Exception as e:
                dbg("禁用右键菜单(非致命): %r" % (e,))
            mh = MsgHandler()
            WEBVIEW_MSG_HANDLER = mh
            ALIVE.append(mh)
            iface = mh.QueryInterface(WV.ICoreWebView2WebMessageReceivedEventHandler)
            webview.add_WebMessageReceived(iface)
            dbg("消息处理器已注册")
            # 用 file:// 加载（支持相对路径引用本地图片，NavigateToString 不支持）
            _file_url = "file:///" + HTML_PATH.replace("\\", "/")
            webview.Navigate(_file_url)
            dbg("Navigate file:// 已调用: %s" % _file_url)
        except Exception as e:
            dbg("控制器初始化异常: %r" % (e,))
            import traceback
            dbg(traceback.format_exc())
            ctypes.windll.user32.MessageBoxW(None,
                u"WebView2 界面加载异常: %r" % (e,),
                u"ADOFAI GUI", 0x10)


class MsgHandler(comtypes.COMObject):
    _com_interfaces_ = [WV.ICoreWebView2WebMessageReceivedEventHandler]
    def Invoke(self, sender, args):
        try:
            s = args.TryGetWebMessageAsString()
            if isinstance(s, tuple):
                s = s[-1]
            if not s:
                return
            handle_web_message(s)
        except Exception:
            pass


# ---------- 宿主 <-> 网页 通信 ----------
def post_to_web(obj):
    if WEBVIEW is None:
        return
    try:
        WEBVIEW.PostWebMessageAsString(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass


def push_log(text):
    with log_lock:
        log_q.append(text)
    if HWND_MAIN:
        user32.PostMessageW(HWND_MAIN, WM_APP_LOG, 0, 0)


def push_done(out, code, error=None):
    global last_out, last_code, last_error
    last_out, last_code, last_error = out, code, error
    if HWND_MAIN:
        user32.PostMessageW(HWND_MAIN, WM_APP_DONE, 0, 0)


def handle_web_message(s):
    try:
        d = json.loads(s)
    except Exception:
        return
    t = d.get("type")
    if t == "pickfile":
        global pick_ctx
        pick_ctx = {"dest": d.get("dest", "")}
        if HWND_MAIN:
            user32.PostMessageW(HWND_MAIN, WM_APP_PICK, 0, 0)
        return
    if t == "generate":
        start_generate(d)
    elif t == "drag":
        if HWND_MAIN:
            user32.PostMessageW(HWND_MAIN, WM_APP_DRAG, 0, 0)
    elif t == "resize":
        if HWND_MAIN:
            ht = EDGE_HT.get(str(d.get("edge", "")))
            if ht is not None:
                user32.PostMessageW(HWND_MAIN, WM_APP_RESIZE, ht, 0)
    elif t == "winminimize":
        if HWND_MAIN:
            user32.ShowWindow(HWND_MAIN, SW_MINIMIZE)
    elif t == "winmaximize":
        if HWND_MAIN:
            user32.ShowWindow(HWND_MAIN,
                              SW_RESTORE if user32.IsZoomed(HWND_MAIN) else SW_MAXIMIZE)
    elif t == "close":
        if HWND_MAIN:
            user32.PostMessageW(HWND_MAIN, WM_CLOSE, 0, 0)
    elif t == "listhistory":
        send_history()
    elif t == "modelstatus":
        send_model_status()
    elif t == "setbackdrop":
        if HWND_MAIN:
            set_backdrop(HWND_MAIN, int(d.get("mat", 2)))
    elif t == "setopaque":
        set_opaque(bool(d.get("on", False)))
    elif t == "openfile":
        p = d.get("path")
        if p and os.path.isfile(p):
            try:
                os.startfile(p)
            except Exception as e:
                dbg("openfile err: %r" % (e,))
    elif t == "openfolder":
        # 打开用户源文件所在目录（非 preview 目录）
        folder = d.get("folder", "")
        if folder and os.path.isdir(folder):
            try:
                os.startfile(folder)
            except Exception as e:
                dbg("openfolder err: %r" % (e,))
        else:
            # 无源文件目录信息时，回退打开 preview 目录
            p = d.get("path", "")
            if p:
                _d = os.path.dirname(p)
                if os.path.isdir(_d):
                    os.startfile(_d)
    elif t == "deletehist":
        # 删除 preview 里的 .adofai 文件（及 .meta.json sidecar），不删源文件
        p = d.get("path")
        if p and os.path.isfile(p):
            try:
                os.remove(p)
                _meta = p + ".meta.json"
                if os.path.isfile(_meta):
                    os.remove(_meta)
                send_history()  # 刷新列表
            except Exception as e:
                dbg("deletehist err: %r" % (e,))
    # ---------- 音源分离 / 试听（#203）----------
    elif t == "separate":
        start_separate(d)
    elif t == "preview":
        start_preview(d)
    # ---------- 训练：踩点/风格（#205）----------
    elif t == "train_start":
        try:
            ok, msg = ws._start_training(
                d.get("kind", "all"),
                d.get("data_dir", ""),
                {"train_target": d.get("target", "melody"),
                 "onset_epochs": int(d.get("onset_epochs", 80)),
                 "vae_epochs": int(d.get("vae_epochs", 0) or 0),
                 "ddpm_epochs": int(d.get("ddpm_epochs", 0) or 0)})
            post_to_web({"type": "train_result", "ok": ok, "msg": msg})
        except Exception as e:
            post_to_web({"type": "train_result", "ok": False, "error": "启动训练失败：%s" % e})
    elif t == "train_stop":
        proc = ws.TRAIN_STATE.get("proc")
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        ws.TRAIN_STATE["status"] = "idle"
        ws.TRAIN_STATE["proc"] = None
        post_to_web({"type": "train_result", "ok": True, "msg": "已停止训练"})
    elif t == "train_log":
        gui_train_log()
    # ---------- 训练：VFX 视觉特效（#204）----------
    elif t == "train_vfx_start":
        try:
            ok, msg = ws._start_vfx_training(
                d.get("data_dir", ""), int(d.get("epochs", 60)),
                bool(d.get("rebuild", False)))
            post_to_web({"type": "vfx_result", "ok": ok, "msg": msg})
        except Exception as e:
            post_to_web({"type": "vfx_result", "ok": False, "error": "启动 VFX 训练失败：%s" % e})
    elif t == "train_vfx_stop":
        proc = ws.VFX_STATE.get("proc")
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        ws.VFX_STATE["status"] = "idle"
        ws.VFX_STATE["proc"] = None
        post_to_web({"type": "vfx_result", "ok": True, "msg": "已停止 VFX 训练"})
    elif t == "vfx_log":
        gui_vfx_log()
    # ---------- 训练：Shape 走线模型（#206）----------
    elif t == "shape_start":
        start_shape_training(d.get("data_dir", ""))
    elif t == "shape_stop":
        stop_shape_training()
    elif t == "shape_log":
        gui_shape_log()


def send_history():
    out_dir = os.path.join(DATA_DIR, "preview")
    items = []
    try:
        os.makedirs(out_dir, exist_ok=True)
        for fn in sorted(os.listdir(out_dir), reverse=True):
            if fn.lower().endswith(".adofai"):
                fp = os.path.join(out_dir, fn)
                sz = os.path.getsize(fp)
                if sz >= 1024 * 1024:
                    size = "%.1f MB" % (sz / 1024.0 / 1024.0)
                else:
                    size = "%.1f KB" % (sz / 1024.0)
                # 读 metadata sidecar（源文件路径）
                source_dir = ""
                meta_path = fp + ".meta.json"
                if os.path.isfile(meta_path):
                    try:
                        import json as _json
                        with open(meta_path, "r", encoding="utf-8") as _mf:
                            _m = _json.load(_mf)
                            _sp = _m.get("source_path", "")
                            if _sp:
                                source_dir = os.path.dirname(_sp)
                    except Exception:
                        pass
                items.append({"name": fn, "size": size, "path": fp, "source_dir": source_dir})
    except Exception as e:
        dbg("send_history err: %r" % (e,))
    post_to_web({"type": "history", "items": items})


def send_model_status():
    # 统一权重解析（运行时目录=用户训练 优先 > 便携内置=出厂），
    # 与推理端 paths.resolve_checkpoint 一致，不再只盯 RUNTIME_DIR 一处。
    from paths import resolve_checkpoint
    models = [
        ("onset_net.pt", "OnsetNet"),
        ("onset_net_melody.pt", "OnsetNet-Melody"),
        ("onset_net_vocal.pt", "OnsetNet-Vocal"),
        ("vae.pt", "VAE"),
        ("ddpm.pt", "DDPM"),
        ("vfx_net.pt", "VFXNet"),
        ("shape_model.pt", "ShapeModel"),
    ]
    items = []
    try:
        for fn, label in models:
            p = resolve_checkpoint(fn)
            ok = p is not None
            sz = round(p.stat().st_size / 1048576, 1) if ok else 0
            items.append({"label": label, "present": ok, "size_mb": sz})
    except Exception as e:
        dbg("send_model_status err: %r" % (e,))
    post_to_web({"type": "modelstatus", "items": items})


def set_opaque(on):
    global WEBVIEW_CTRL
    if WEBVIEW_CTRL is None:
        return
    try:
        ctrl2 = WEBVIEW_CTRL.QueryInterface(WV.ICoreWebView2Controller2)
        if on:
            # 不透明深色，规避透明渲染 bug
            ctrl2.DefaultBackgroundColor = WV.COREWEBVIEW2_COLOR(A=255, R=20, G=20, B=24)
        else:
            ctrl2.DefaultBackgroundColor = WV.COREWEBVIEW2_COLOR(A=0, R=0, G=0, B=0)
        dbg("set_opaque(%s) OK" % on)
    except Exception as e:
        dbg("set_opaque err: %r" % (e,))


def start_generate(params):
    global gen_thread
    audio = (params.get("audio") or "").strip()
    if not audio or not os.path.isfile(audio):
        post_to_web({"type": "log", "text": "[gui] 请先选择有效的音频文件"})
        post_to_web({"type": "state", "busy": False})
        return
    if gen_thread and gen_thread.is_alive():
        post_to_web({"type": "log", "text": "[gui] 已有生成任务在运行，请稍候"})
        return
    post_to_web({"type": "state", "busy": True})
    gen_thread = threading.Thread(target=run_generate, args=(params, audio), daemon=True)
    gen_thread.start()


def run_generate(params, audio):
    stem = os.path.splitext(os.path.basename(audio))[0]
    out = os.path.join(DATA_DIR, "preview", stem + ".adofai")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        with open(audio, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        push_log(f"[gui] 读取音频失败: {e}")
        push_done(None, -1, "读取音频失败: %s" % e)
        return

    bpm_raw = params.get("bpm")
    if params.get("auto_bpm"):
        bpm_raw = ""
    track = (params.get("track") or "all")
    selected = [track] if track and track != "all" else []
    gen_params = {
        "bpm": bpm_raw if (bpm_raw not in (None, "")) else "",
        "difficulty": int(params.get("difficulty") or 1),
        "selected": selected,
        "track": "all",
        "vfx": bool(params.get("vfx")),
        "intensity": 0.5,                       # 不暴露给用户（特效自由发挥）
        "onset_mode": params.get("mode") or "auto",
        "dual_block": bool(params.get("dual_block", True)),   # 同音多采拦截开关（界面勾选）
    }
    try:
        from web_server import generate_chart_model
    except Exception as e:
        push_log(f"[gui] 加载生成模块失败: {e}")
        push_done(None, -1, "加载生成模块失败: %s" % e)
        return

    push_log(f"[gui] 开始生成: {os.path.basename(audio)} | 音轨={track} | 难度={gen_params['difficulty']} | 模式={gen_params['onset_mode']}")
    try:
        result = generate_chart_model(audio_bytes, os.path.basename(audio), gen_params)
    except Exception as e:
        push_log(f"[gui] 生成异常: {e}")
        push_done(None, -1, "生成异常: %s" % e)
        return

    if not result.get("ok"):
        push_log(f"[gui] 生成失败: {result.get('error')}")
        push_done(None, -1, result.get('error') or "生成失败（原因未知）")
        return

    try:
        with open(out, "w", encoding="utf-8-sig") as f:
            f.write(result.get("adofai", ""))
    except Exception as e:
        push_log(f"[gui] 写谱面文件失败: {e}")

    # 再复制一份到原输入音频所在目录（便于直接丢进游戏关卡文件夹）。
    audio_dir = os.path.dirname(audio)
    if audio_dir and os.path.isdir(audio_dir):
        try:
            copy_out = os.path.join(audio_dir, stem + ".adofai")
            if os.path.abspath(copy_out) != os.path.abspath(out):
                with open(copy_out, "w", encoding="utf-8-sig") as f:
                    f.write(result.get("adofai", ""))
                push_log(f"[gui] 已复制到原音频目录: {copy_out}")
        except Exception as e:
            push_log(f"[gui] 复制到原音频目录失败（不影响 preview 副本）: {e}")

    post_to_web({
        "type": "result", "ok": True,
        "png": result.get("preview_png"),
        "note_count": result.get("note_count"),
        "twirl_count": result.get("twirl_count"),
        "vfx_count": result.get("vfx_count"),
        "duration_s": result.get("duration_s"),
        "bpm_used": result.get("bpm_used"),
        "out": out,
    })
    push_log(f"[gui] 完成！音符 {result.get('note_count')} | twirl {result.get('twirl_count')} | vfx {result.get('vfx_count')} | {result.get('duration_s')}s | BPM {result.get('bpm_used')}")
    push_done(out, 0)


# ---------- 音源分离 / 试听（#203）----------
def _read_audio_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        return None, str(e)


def start_separate(params):
    audio = (params.get("audio") or "").strip()
    if not audio or not os.path.isfile(audio):
        post_to_web({"type": "separate_result", "ok": False, "error": "请先选择有效的音频文件"})
        return
    post_to_web({"type": "separate_status", "text": "分解中（首次约 5–20 秒）…"})
    try:
        data = _read_audio_bytes(audio)
        if data is None:
            post_to_web({"type": "separate_result", "ok": False, "error": "读取音频失败"})
            return
        result = ws.separate_all(data, os.path.basename(audio))
    except Exception as e:
        post_to_web({"type": "separate_result", "ok": False, "error": "分离异常：%s" % e})
        return
    if not result.get("ok"):
        post_to_web({"type": "separate_result", "ok": False, "error": result.get("error", "分离失败")})
        return
    # 把 web 版 /preview/<file> url 改写成 file:// 绝对路径（GUI 用 file:// 导航，虚拟主机不生效）
    preview_root = "file:///" + ws.PREVIEW_DIR.replace("\\", "/") + "/"
    stems = []
    for s in result.get("stems", []):
        u = s.get("url", "")
        if u.startswith("/preview/"):
            u = preview_root + u[len("/preview/"):]
        stems.append({"name": s.get("name"), "label": s.get("label"),
                      "url": u, "duration_s": s.get("duration_s")})
    post_to_web({"type": "separate_result", "ok": True, "stems": stems})


def start_preview(params):
    audio = (params.get("audio") or "").strip()
    track = (params.get("track") or "all")
    if not audio or not os.path.isfile(audio):
        post_to_web({"type": "preview_result", "ok": False, "error": "请先选择有效的音频文件"})
        return
    post_to_web({"type": "preview_status", "text": "分离「%s」试听中…" % track})
    try:
        data = _read_audio_bytes(audio)
        if data is None:
            post_to_web({"type": "preview_result", "ok": False, "error": "读取音频失败"})
            return
        result = ws.preview_track(data, os.path.basename(audio), track)
    except Exception as e:
        post_to_web({"type": "preview_result", "ok": False, "error": "试听异常：%s" % e})
        return
    if not result.get("ok"):
        post_to_web({"type": "preview_result", "ok": False, "error": result.get("error", "试听失败")})
        return
    u = result.get("url", "")
    if u.startswith("/preview/"):
        u = "file:///" + ws.PREVIEW_DIR.replace("\\", "/") + "/" + u[len("/preview/"):]
    post_to_web({"type": "preview_result", "ok": True, "url": u, "track": track,
                 "duration_s": result.get("duration_s")})


# ---------- 训练日志读取（轮询用）----------
def _tail(path, n=400):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def gui_train_log():
    log = _tail(ws.TRAIN_LOG, 400)
    st = ws.TRAIN_STATE["status"]
    running = (st == "running") or (ws.TRAIN_STATE.get("proc") is not None)
    post_to_web({"type": "train_log", "log": log, "status": st,
                 "running": running, "kind": ws.TRAIN_STATE.get("kind")})


def gui_vfx_log():
    log = _tail(ws.VFX_LOG, 400)
    st = ws.VFX_STATE["status"]
    running = (st == "running") or (ws.VFX_STATE.get("proc") is not None)
    post_to_web({"type": "vfx_log", "log": log, "status": st,
                 "running": running, "kind": ws.VFX_STATE.get("kind")})


def gui_shape_log():
    """ShapeModel 训练实时监控（读 data/shape_train.log + 预处理进度）。"""
    import glob as _glob
    log = _tail(SHAPE_LOG, 600)
    lines = log.splitlines()
    skip = sum(1 for l in lines if "[skip]" in l)
    ok = sum(1 for l in lines if l.strip().startswith("+ "))
    feat_dir = os.path.join(DATA_DIR, "shape_feat_cache")
    feat_done = len(_glob.glob(os.path.join(feat_dir, "*.npz"))) if os.path.isdir(feat_dir) else 0
    running = (SHAPE_STATE["status"] == "running") or (SHAPE_STATE.get("proc") is not None)
    if not running and os.path.exists(SHAPE_LOG):
        running = (time.time() - os.path.getmtime(SHAPE_LOG)) < 120
    ckpt = os.path.join(DATA_DIR, "checkpoints", "shape_model.pt")
    done = os.path.exists(ckpt)
    post_to_web({"type": "shape_log", "log": log, "skip": skip, "ok": ok,
                 "feat_done": feat_done, "feat_total": 105,
                 "running": running, "done": done})


def start_shape_training(data_dir):
    if SHAPE_STATE.get("proc") is not None:
        post_to_web({"type": "shape_result", "ok": False, "error": "Shape 训练正在进行中，请先停止"})
        return
    if not data_dir or not os.path.isdir(data_dir):
        post_to_web({"type": "shape_result", "ok": False,
                     "error": "训练数据目录不存在：%s" % data_dir})
        return
    if not os.path.exists(TRAIN_SHAPE_SCRIPT):
        post_to_web({"type": "shape_result", "ok": False,
                     "error": "未找到训练脚本：%s" % TRAIN_SHAPE_SCRIPT})
        return
    try:
        os.makedirs(os.path.dirname(SHAPE_LOG), exist_ok=True)
        open(SHAPE_LOG, "w").close()
    except Exception:
        pass
    env = dict(os.environ)
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8",
        "MKL_THREADING_LAYER": "sequential", "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "KMP_AFFINITY": "disabled",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "ADOFAI_DATA_DIR": DATA_DIR,
        "PYTHONPATH": VENV_SP,
    })
    try:
        logf = open(SHAPE_LOG, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [ws.VENV_PY, TRAIN_SHAPE_SCRIPT, "--data", data_dir],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
    except Exception as e:
        try:
            logf.close()
        except Exception:
            pass
        post_to_web({"type": "shape_result", "ok": False, "error": "启动失败：%s" % e})
        return
    SHAPE_STATE["proc"] = proc
    SHAPE_STATE["status"] = "running"
    SHAPE_STATE["started"] = time.time()

    def _watch():
        try:
            proc.wait()
        finally:
            try:
                logf.close()
            except Exception:
                pass
            SHAPE_STATE["proc"] = None
            SHAPE_STATE["status"] = "done"
    threading.Thread(target=_watch, daemon=True).start()
    post_to_web({"type": "shape_result", "ok": True, "msg": "已启动 Shape 训练（监控页可看实时进度，停止按钮可中断）"})


def stop_shape_training():
    proc = SHAPE_STATE.get("proc")
    if proc:
        try:
            proc.kill()
        except Exception:
            pass
    SHAPE_STATE["proc"] = None
    SHAPE_STATE["status"] = "idle"
    post_to_web({"type": "shape_result", "ok": True, "msg": "已停止 Shape 训练"})


# ---------- 窗口过程 ----------
def wndproc(hwnd, msg, wparam, lparam):
    global HWND_MAIN, last_zoomed
    if msg == WM_ERASEBKGND:
        return 1  # 背景由 DWM Mica 提供，禁止白底擦除
    elif msg == WM_CONTEXTMENU:
        return 0  # 禁止窗口内右键（不弹任何菜单）
    elif msg == WM_SIZE:
        w = lparam & 0xFFFF
        h = (lparam >> 16) & 0xFFFF
        if WEBVIEW_CTRL is not None:
            try:
                WEBVIEW_CTRL.Bounds = wt.RECT(0, 0, w, h)
            except Exception:
                pass
        try:
            zoomed = bool(user32.IsZoomed(hwnd))
            if zoomed != last_zoomed:
                last_zoomed = zoomed
                post_to_web({"type": "winstate", "zoomed": zoomed})
        except Exception:
            pass
        return 0
    elif msg == WM_DESTROY:
        try:
            if WEBVIEW_CTRL is not None:
                WEBVIEW_CTRL.Close()
        except Exception:
            pass
        user32.PostQuitMessage(0)
        return 0
    elif msg == WM_CLOSE:
        user32.DestroyWindow(hwnd)
        return 0
    elif msg == WM_APP_LOG:
        with log_lock:
            text = log_q.popleft() if log_q else ""
        if text:
            post_to_web({"type": "log", "text": text})
        return 0
    elif msg == WM_APP_DONE:
        post_to_web({"type": "done", "out": last_out, "code": last_code, "error": last_error})
        post_to_web({"type": "state", "busy": False})  # 解灰生成按钮
        return 0
    elif msg == WM_APP_DRAG:
        user32.ReleaseCapture()
        pt = wt.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        lp = (pt.y << 16) | pt.x
        user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, lp)
        return 0
    elif msg == WM_APP_PICK:
        path = native_open_file()
        if path:
            post_to_web({"type": "filepath", "path": path,
                         "dest": getattr(pick_ctx, "dest", "")})
        return 0
    elif msg == WM_APP_RESIZE:
        # 由 HTML 边缘热区发起的标准 Windows 缩放（WebView2 铺满时父窗口收不到边缘命中）
        user32.ReleaseCapture()
        pt = wt.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        lp = (pt.y << 16) | pt.x
        user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, wparam, lp)
        return 0
    elif msg == WM_NCCALCSIZE:
        if wparam:
            return 0   # 客户区填满窗口（无边框）
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
    elif msg == WM_NCHITTEST:
        res = user32.DefWindowProcW(hwnd, msg, wparam, lparam)
        if res != HTCLIENT:
            return res
        if user32.IsZoomed(hwnd):
            return HTCLIENT
        x = lparam & 0xFFFF
        y = (lparam >> 16) & 0xFFFF
        r = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        cx = x - r.left; cy = y - r.top
        W = r.right - r.left; H = r.bottom - r.top
        b = RESIZE_BORDER
        if cx <= b and cy <= b: return HTTOPLEFT
        if cx >= W - b and cy <= b: return HTTOPRIGHT
        if cx <= b and cy >= H - b: return HTBOTTOMLEFT
        if cx >= W - b and cy >= H - b: return HTBOTTOMRIGHT
        if cx <= b: return HTLEFT
        if cx >= W - b: return HTRIGHT
        if cy <= b: return HTTOP
        if cy >= H - b: return HTBOTTOM
        return HTCLIENT
    elif msg == WM_SETCURSOR:
        if user32.LOWORD(lparam) == HTCLIENT:
            # 查询当前命中结果，设置对应光标
            pt = wt.POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            ht = user32.SendMessageW(hwnd, WM_NCHITTEST, 0,
                                     (pt.y << 16) | pt.x) & 0xFFFF
            cursors = {
                HTLEFT: IDC_SIZEWE, HTRIGHT: IDC_SIZEWE,
                HTTOP: IDC_SIZENS, HTBOTTOM: IDC_SIZENS,
                HTTOPLEFT: IDC_SIZENWSE, HTBOTTOMRIGHT: IDC_SIZENWSE,
                HTTOPRIGHT: IDC_SIZENESW, HTBOTTOMLEFT: IDC_SIZENESW,
            }
            cid = cursors.get(ht, 0)
            if cid:
                user32.SetCursor(user32.LoadCursorW(None, ctypes.cast(cid, wt.LPCWSTR)))
                return 1
    elif msg == WM_GETMINMAXINFO:
        mmi = ctypes.cast(ctypes.c_void_p(lparam), ctypes.POINTER(MINMAXINFO)).contents
        mmi.ptMinTrackSize = wt.POINT(WIN_MIN_W, WIN_MIN_H)
        return 0
    elif msg == WM_NCLBUTTONDOWN:
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


# ---------- 入口 ----------
def main():
    global HWND_MAIN
    # 线程已在 import 前初始化为 STA（WebView2 必需）。这里再确认一次。
    try:
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        ole32.CoInitializeEx.restype = ctypes.HRESULT
        _r = ole32.CoInitializeEx(None, 2)
        dbg("main: CoInitializeEx(STA) 再次调用 hr=%s" % hex(_r & 0xFFFFFFFF))
    except Exception as e:
        dbg("main: CoInitializeEx 异常: %r" % (e,))
    hinst = kernel32.GetModuleHandleW(None)
    wproc = WNDPROC(wndproc)

    # 加载自定义图标（从 ICO 文件，带 alpha 透明，无白边）
    icon_ico = os.path.join(ROOT, "gui", "icon.ico")
    hIconLarge = None
    hIconSmall = None
    if os.path.isfile(icon_ico):
        # 指定尺寸，从多尺寸 ICO 里挑对应帧（带 alpha 通道）
        hIconLarge = user32.LoadImageW(None, wt.LPCWSTR(icon_ico), IMAGE_ICON, 32, 32,
                                        LR_LOADFROMFILE)
        hIconSmall = user32.LoadImageW(None, wt.LPCWSTR(icon_ico), IMAGE_ICON, 16, 16,
                                        LR_LOADFROMFILE)
        dbg(f"LoadImageW icon (ico) -> large={hex(hIconLarge) if hIconLarge else 'None'} "
            f"small={hex(hIconSmall) if hIconSmall else 'None'}")

    wc = WNDCLASSEX()
    wc.cbSize = ctypes.sizeof(WNDCLASSEX)
    wc.style = CS_HREDRAW | CS_VREDRAW
    wc.lpfnWndProc = wproc
    wc.hInstance = hinst
    wc.hCursor = user32.LoadCursorW(None, ctypes.cast(32512, wt.LPCWSTR))  # IDC_ARROW
    wc.hIcon = wt.HICON(hIconLarge or 0)   # 大图标（任务栏/Alt+Tab）
    wc.hIconSm = wt.HICON(hIconSmall or 0)  # 小图标（标题栏左上角）
    wc.hbrBackground = None  # NULL：不画背景，露出 Mica
    wc.lpszClassName = wt.LPCWSTR(WND_CLASS)
    atom = user32.RegisterClassExW(ctypes.byref(wc))
    if not atom:
        ctypes.windll.user32.MessageBoxW(None,
            "RegisterClassExW 失败", "ADOFAI GUI", 0x10)
        return

    hwnd = user32.CreateWindowExW(0, wt.LPCWSTR(WND_CLASS),
        wt.LPCWSTR("ADOFAI 谱面生成器"),
        WS_POPUP | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, WIN_W, WIN_H,
        None, None, hinst, None)
    if not hwnd:
        ctypes.windll.user32.MessageBoxW(None,
            "CreateWindowExW 失败（句柄为 0，检查 ctypes argtypes）", "ADOFAI GUI", 0x10)
        return
    HWND_MAIN = hwnd

    # 设置窗口图标（保险：有些场景下 WNDCLASSEX 的图标不够）
    if hIconLarge:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hIconLarge)
    if hIconSmall:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hIconSmall)
    set_backdrop(hwnd, 3)                     # Acrylic 透明背景（默认；删除 extend_frame 后失焦也保持透明，不会退化成灰白）
    set_round(hwnd)                          # 圆角
    set_dark(hwnd)                           # 深色标题栏（若有）
    user32.ShowWindow(hwnd, 1)
    # 清掉系统窗口图标（标题栏左侧不再显示 Windows 默认图标，用 HTML 内联 SVG 替代）
    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, 0)
    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, 0)
    user32.UpdateWindow(hwnd)

    # 启动 WebView2
    try:
        a = ctypes.c_int(); b = ctypes.c_int()
        if ole32.CoGetApartmentType:
            ole32.CoGetApartmentType.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            ole32.CoGetApartmentType.restype = ctypes.HRESULT
            ole32.CoGetApartmentType(ctypes.byref(a), ctypes.byref(b))
        dbg("main: COM 套间 aptType=%s qual=%s" % (a.value, b.value))
    except Exception as e:
        dbg("CoGetApartmentType 异常: %r" % (e,))
    dbg("开始创建 WebView2 环境, loader=%s" % LOADER)
    os.makedirs(WEBVIEW_CACHE, exist_ok=True)
    env_handler = EnvCompleted()
    ALIVE.append(env_handler)
    iface = env_handler.QueryInterface(WV.ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler)
    hr = _create_env(None, WEBVIEW_CACHE, None, iface)
    rh = hex(hr & 0xFFFFFFFF) if isinstance(hr, int) else hr
    dbg("CreateCoreWebView2EnvironmentWithOptions hr=%s" % rh)
    if hr < 0:
        ctypes.windll.user32.MessageBoxW(hwnd,
            u"WebView2 初始化失败\nHRESULT=%s\n请确认本机已安装 WebView2 Runtime" % rh,
            u"ADOFAI GUI", 0x10)

    # 看门狗：5 秒后若网页仍未就绪，提示去看日志（避免无声黑屏）
    def _watchdog():
        import time
        time.sleep(5)
        if WEBVIEW is None:
            dbg("WATCHDOG: 5 秒后 WebView 仍未就绪")
            ctypes.windll.user32.MessageBoxW(HWND_MAIN,
                u"WebView2 未能在 5 秒内就绪，界面可能黑屏。\n请打开 gui 目录下的 gui_debug.log 查看失败原因，并把内容发给我。",
                u"ADOFAI GUI", 0x10)
    threading.Thread(target=_watchdog, daemon=True).start()

    msg = wt.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            ctypes.windll.user32.MessageBoxW(None, tb, "ADOFAI GUI 崩溃", 0x10)
        except Exception:
            pass
