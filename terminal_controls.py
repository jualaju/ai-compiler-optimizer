"""Shared, testable raw-key navigation for both terminal debugging dashboards."""
from __future__ import annotations
import os
import select
import sys
from rich.console import Group
from rich.live import Live
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel


def choice(console, label, choices, default):
    while True:
        try: value = console.input(f'{label} ({default}): ').strip() or default
        except EOFError: value = default
        if value in choices: return value
        console.print(f'Invalid selection {value!r}. Choose: {", ".join(choices)}.', style='yellow', markup=False)


def navigate(key, mode, offset, views, home_key, up_key):
    if key in ('q', ''): return mode, offset, 'quit'
    if key == home_key: return 'overview', 0, ''
    if key in views: return key, 0, ''
    if key == 'n': return mode, offset+1, '' if mode != 'overview' else 'Open a detail view before scrolling.'
    if key == up_key: return mode, max(0,offset-1), '' if mode != 'overview' else 'Open a detail view before scrolling.'
    if key == '?': return mode, offset, 'Keys: ' + ', '.join(views) + f', {home_key} overview, N/{up_key.upper()} scroll, Q quit.'
    return mode, offset, f'Unknown key {key!r}. Press ? for available keys.'


def browse(console, overview, views, *, titles=None, initial='overview', home_key='a', up_key='u', syntax_views=()):
    """Call view suppliers on navigation; failed actions retain the last good view."""
    import termios
    import tty
    mode, offset, message = initial, 0, ''
    cached = {}
    if initial != 'overview': cached[initial] = views[initial]()
    settings = termios.tcgetattr(sys.stdin.fileno())
    help_text = ' · '.join(f'{key.upper()} {title}' for key,title in (titles or {}).items())
    help_text += f' · {home_key.upper()} overview · N/{up_key.upper()} scroll · ? help · Q quit'
    last = None
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                current = (mode, offset, message, console.size)
                if current != last:
                    if mode == 'overview':
                        view = overview()
                        # Layout dashboards keep their own footer; feedback gets its own view.
                        if message: view = Group(Panel(Text(message,style='yellow')), view)
                    else:
                        lines = Text(cached.get(mode, '')).wrap(console, max(8,console.width-4))
                        page = max(1,console.height-7)
                        offset = min(offset,max(0,len(lines)-page))
                        body = '\n'.join(line.plain for line in lines[offset:offset+page])
                        content = Syntax(body,'python',theme='ansi_dark',word_wrap=True) if mode in syntax_views else Text(body)
                        view = Group(Panel(content,title=Text((titles or {}).get(mode,mode).upper()),border_style='cyan'),
                                     Text(message or f'Lines {offset+1}–{min(offset+page,len(lines))}/{len(lines)}',style='yellow' if message else 'dim'),
                                     Text(help_text,style='cyan'))
                    live.update(view,refresh=True)
                    last = (mode,offset,message,console.size)
                if not select.select([sys.stdin],[],[],.25)[0]: continue
                key = os.read(sys.stdin.fileno(),1).decode(errors='replace').lower()
                next_mode,next_offset,next_message = navigate(key,mode,offset,views,home_key,up_key)
                if next_message == 'quit': break
                if key in views:
                    try:
                        live.update(Panel(Text(f'Opening {key.upper()} — waiting for the requested operation…')),refresh=True)
                        cached[key] = views[key]()
                    except Exception as exc:
                        message = f'Action {key.upper()} failed: {exc}. Previous results retained; choose another view or retry.'
                        last = None
                        continue
                mode,offset,message = next_mode,next_offset,next_message
    finally:
        termios.tcsetattr(sys.stdin.fileno(),termios.TCSADRAIN,settings)
    console.print(overview())
