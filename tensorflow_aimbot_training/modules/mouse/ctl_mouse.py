
class ctlmouse:
  def click(x,y):
    ctypes.windll.user32.SetCursorPos(x, y)
    #ctypes.windll.user32.mouse_event(2, 0, 0, 0,0) # left down
    #ctypes.windll.user32.mouse_event(4, 0, 0, 0,0) # left up

  class _point_t(ctypes.Structure):
      _fields_ = [
                  ('x',  ctypes.c_long),
                  ('y',  ctypes.c_long),
                 ]
                 
  def mouse_move_abs(x,y):
    point = _point_t()
    result = ctypes.windll.user32.GetCursorPos(ctypes.pointer(point))
    if result:
      ctypes.windll.user32.SetCursorPos(point.x + x, point.y + y)