\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2026/04/20 v1.1 Klasa pracy dyplomowej PJATK]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt]{report}

\RequirePackage{mathptmx}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[polish]{babel}
\RequirePackage{polski}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}
\RequirePackage{ifthen}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage[table]{xcolor}
\RequirePackage{hyperref}
\RequirePackage{indentfirst}
\RequirePackage{caption}
\RequirePackage{booktabs}
\RequirePackage{titlesec}
\RequirePackage[backend=biber, style=ieee]{biblatex}
\RequirePackage{footmisc}
\RequirePackage{tikz}
\RequirePackage{adjustbox}
\RequirePackage{float}
\RequirePackage{tabularx}
\RequirePackage{csquotes}
\RequirePackage{microtype}
\RequirePackage{rotating}
\RequirePackage{siunitx}
\RequirePackage{multirow}
\RequirePackage{listings}
\RequirePackage{lstautogobble}
\RequirePackage[htt]{hyphenat}
\RequirePackage[nameinlink]{cleveref}

\AtBeginDocument{%
  \let\origUNDERSCORE\_%
  \renewcommand{\_}{\origUNDERSCORE\penalty0\hskip0pt\relax}%
}

\crefname{chapter}{rozdziale}{rozdzialach}
\Crefname{chapter}{Rozdzia\l{}}{Rozdzia\l{}y}
\crefname{section}{sekcji}{sekcjach}
\Crefname{section}{Sekcja}{Sekcje}
\crefname{subsection}{podsekcji}{podsekcjach}
\Crefname{subsection}{Podsekcja}{Podsekcje}
\crefname{figure}{rysunku}{rysunkach}
\Crefname{figure}{Rysunek}{Rysunki}
\crefname{table}{tabeli}{tabelach}
\Crefname{table}{Tabela}{Tabele}
\crefname{equation}{r\'ownaniu}{r\'ownaniach}
\Crefname{equation}{R\'ownanie}{R\'ownania}
\crefname{listing}{listingu}{listingach}
\Crefname{listing}{Listing}{Listingi}

\newcommand{\crefrangeconjunction}{ do~}
\newcommand{\crefpairconjunction}{ i~}
\newcommand{\crefmiddleconjunction}{, }
\newcommand{\creflastconjunction}{ i~}

\definecolor{pyKeyword}{HTML}{0033B3}
\definecolor{pyBuiltin}{HTML}{267F99}
\definecolor{pyConstant}{HTML}{8B008B}
\definecolor{pySelf}{HTML}{C75A11}
\definecolor{pyDecorator}{HTML}{8C5E00}
\definecolor{pyMagic}{HTML}{0086B3}
\definecolor{pyString}{HTML}{067D17}
\definecolor{pyDocstring}{HTML}{8E908C}
\definecolor{pyNumber}{HTML}{1750EB}
\definecolor{pyComment}{HTML}{8C8C8C}
\definecolor{pyOperator}{HTML}{555555}
\definecolor{pyFrame}{HTML}{D0D0D0}
\definecolor{pyLineNo}{HTML}{999999}
\definecolor{pyBg}{HTML}{FBFBFB}

\lstset{
  language=Python,
  backgroundcolor=\color{pyBg},
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  breakatwhitespace=true,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  alsoletter={@,_},
  sensitive=true,
  commentstyle=\color{pyComment}\itshape,
  keywordstyle=\color{pyKeyword}\bfseries,
  keywordstyle=[2]\color{pyConstant}\bfseries,
  keywordstyle=[3]\color{pyBuiltin},
  keywordstyle=[4]\color{pySelf}\itshape,
  keywordstyle=[5]\color{pyDecorator}\bfseries,
  keywordstyle=[6]\color{pyMagic}\itshape,
  stringstyle=\color{pyString},
  identifierstyle=\color{black},
  morekeywords={
    and, as, assert, async, await, break, class, continue, def, del, elif,
    else, except, finally, for, from, global, if, import, in, is, lambda,
    match, case, nonlocal, not, or, pass, raise, return, try, while, with, yield
  },
  morekeywords=[2]{True, False, None, NotImplemented, Ellipsis},
  morekeywords=[3]{
    abs, all, any, ascii, bin, bool, breakpoint, bytearray, bytes, callable,
    chr, classmethod, compile, complex, delattr, dict, dir, divmod, enumerate,
    eval, exec, filter, float, format, frozenset, getattr, globals, hasattr,
    hash, help, hex, id, input, int, isinstance, issubclass, iter, len, list,
    locals, map, max, memoryview, min, next, object, oct, open, ord, pow,
    print, property, range, repr, reversed, round, set, setattr, slice,
    sorted, staticmethod, str, sum, super, tuple, type, vars, zip,
    Any, Optional, Union, Callable, Iterable, Iterator, Generator, Awaitable,
    Coroutine, Sequence, Mapping, MutableMapping, MutableSequence, List, Dict,
    Tuple, Set, FrozenSet, Type, TypeVar, Literal, Final, Annotated, Protocol,
    NewType, ClassVar, Self, TYPE_CHECKING
  },
  morekeywords=[4]{self, cls},
  morekeywords=[5]{
    @property, @staticmethod, @classmethod, @abstractmethod, @dataclass,
    @cached_property, @contextmanager, @asynccontextmanager, @lru_cache,
    @wraps, @overload, @final, @runtime_checkable, @retry
  },
  morekeywords=[6]{
    __init__, __new__, __del__, __repr__, __str__, __bytes__, __format__,
    __lt__, __le__, __eq__, __ne__, __gt__, __ge__, __hash__, __bool__,
    __getattr__, __getattribute__, __setattr__, __delattr__, __dir__,
    __get__, __set__, __delete__, __set_name__, __slots__, __init_subclass__,
    __class_getitem__, __call__, __len__, __length_hint__, __getitem__,
    __setitem__, __delitem__, __missing__, __iter__, __next__, __reversed__,
    __contains__, __add__, __radd__, __iadd__, __sub__, __mul__, __truediv__,
    __floordiv__, __mod__, __divmod__, __pow__, __enter__, __exit__,
    __aenter__, __aexit__, __aiter__, __anext__, __await__,
    __name__, __doc__, __dict__, __module__, __qualname__, __annotations__,
    __all__, __file__, __version__, __main__
  },
  morestring=[b]{"},
  morestring=[b]{'},
  morecomment=[s][\color{pyDocstring}\itshape]{"""}{"""},
  morecomment=[s][\color{pyDocstring}\itshape]{'''}{'''},
  morecomment=[l][\color{pyComment}\itshape]{\#},
  literate=
  {0}{{{\color{pyNumber}0}}}1 {1}{{{\color{pyNumber}1}}}1
  {2}{{{\color{pyNumber}2}}}1 {3}{{{\color{pyNumber}3}}}1
  {4}{{{\color{pyNumber}4}}}1 {5}{{{\color{pyNumber}5}}}1
  {6}{{{\color{pyNumber}6}}}1 {7}{{{\color{pyNumber}7}}}1
  {8}{{{\color{pyNumber}8}}}1 {9}{{{\color{pyNumber}9}}}1
  {(}{{{\color{pyOperator}(}}}1 {)}{{{\color{pyOperator})}}}1
  {[}{{{\color{pyOperator}[}}}1 {]}{{{\color{pyOperator}]}}}1
  {\{}{{{\color{pyOperator}\{}}}1 {\}}{{{\color{pyOperator}\}}}}1
  {:}{{{\color{pyOperator}:}}}1 {,}{{{\color{pyOperator},}}}1
  {;}{{{\color{pyOperator};}}}1 {=}{{{\color{pyOperator}=}}}1
  {+}{{{\color{pyOperator}+}}}1 {-}{{{\color{pyOperator}-}}}1
  {*}{{{\color{pyOperator}*}}}1 {/}{{{\color{pyOperator}/}}}1
  {<}{{{\color{pyOperator}<}}}1 {>}{{{\color{pyOperator}>}}}1
  {!}{{{\color{pyOperator}!}}}1 {|}{{{\color{pyOperator}|}}}1
  {\&}{{{\color{pyOperator}\&}}}1 {\^{}}{{{\color{pyOperator}\^{}}}}1
  {\~{}}{{{\color{pyOperator}\~{}}}}1 {.}{{{\color{pyOperator}.}}}1
  {ą}{{\k{a}}}1 {ę}{{\k{e}}}1 {ó}{{\'o}}1 {ś}{{\'s}}1
  {ł}{{\l{}}}1 {ż}{{\.z}}1 {ź}{{\'z}}1 {ć}{{\'c}}1 {ń}{{\'n}}1
  {Ą}{{\k{A}}}1 {Ę}{{\k{E}}}1 {Ó}{{\'O}}1 {Ś}{{\'S}}1
  {Ł}{{\L{}}}1 {Ż}{{\.Z}}1 {Ź}{{\'Z}}1 {Ć}{{\'C}}1 {Ń}{{\'N}}1,
  frame=single,
  framerule=0.4pt,
  rulecolor=\color{pyFrame},
  captionpos=t,
  abovecaptionskip=0pt,
  belowcaptionskip=4pt,
  numbers=left,
  numberstyle=\tiny\color{pyLineNo},
  numbersep=8pt,
  xleftmargin=15pt,
  aboveskip=10pt,
  belowskip=6pt,
}

\usetikzlibrary{fit, positioning, arrows.meta, decorations.pathreplacing, shapes, calc}

\newcolumntype{L}{>{\raggedright\arraybackslash}X}

\onehalfspacing

\titleformat{\chapter}[block]
{\normalfont\fontsize{14}{17}\bfseries}
{\thechapter.}{0.5em}{}
\titlespacing*{\chapter}{0pt}{0pt}{20pt}

\titleformat{\section}[block]
{\normalfont\fontsize{12}{14}\bfseries}
{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{14pt}{8pt}

\titleformat{\subsection}[block]
{\normalfont\fontsize{12}{14}\bfseries}
{\thesubsection}{0.5em}{}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}

\captionsetup{
  skip=4pt,
  belowskip=0pt,
  labelfont=bf,
  textfont=bf
}

\captionsetup[table]{
  position=above,
  labelsep=space,
  justification=raggedright,
  singlelinecheck=false,
  name=Tabela,
  labelfont=bf,
  textfont=bf,
  skip=4pt,
  belowskip=0pt
}

\captionsetup[figure]{
  position=above,
  labelsep=space,
  justification=raggedright,
  singlelinecheck=false,
  name=Rysunek,
  labelfont=bf,
  textfont=bf,
  skip=4pt,
  belowskip=0pt
}

\captionsetup[lstlisting]{
  position=above,
  labelsep=space,
  justification=raggedright,
  singlelinecheck=false,
  labelfont=bf,
  textfont=bf,
  skip=4pt,
  belowskip=0pt
}

\floatstyle{plaintop}
\newfloat{wykres}{tbp}{low}[chapter]
\floatname{wykres}{Wykres}
\captionsetup[wykres]{
  position=above,
  labelsep=space,
  justification=raggedright,
  singlelinecheck=false,
  name=Wykres,
  labelfont=bf,
  textfont=bf,
  skip=4pt,
  belowskip=0pt
}

\newcommand{\zrodlo}[1]{%
  \par\vspace{4pt}%
  \begin{minipage}{\linewidth}%
    \raggedright\normalsize\'Zr\'od\l{}o: #1.%
  \end{minipage}%
  \vspace{4pt}%
}

\newenvironment{alwayssingle}{
  \@restonecolfalse
  \if@twocolumn\@restonecoltrue\onecolumn
  \else\if@openright\cleardoublepage\else\clearpage\fi
\fi}%
{\if@restonecol\twocolumn
\else\newpage\thispagestyle{empty}\fi}

\newenvironment{romanpages}
{\cleardoublepage\setcounter{page}{1}%
\renewcommand{\thepage}{\roman{page}}}
{\cleardoublepage\renewcommand{\thepage}{\arabic{page}}%
\setcounter{page}{1}}

\renewcommand{\arraystretch}{1.3}

\renewenvironment{abstract}{%
  \begin{alwayssingle}\thispagestyle{plain}
    \begin{center}
      \vspace*{1.5cm}
      {\large\bfseries Streszczenie}
    \end{center}
    \vspace{0.5cm}
  \begin{quote}}
    {
    \end{quote}
\end{alwayssingle}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  pdfborder={0 0 0}
}