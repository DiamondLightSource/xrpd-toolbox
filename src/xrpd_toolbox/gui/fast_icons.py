from PyQt6.QtCore import QFileInfo
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileIconProvider,
    QStyle,
)


class FastIconProvider(QFileIconProvider):
    """Cheap folder-vs-file icons for QFileSystemModel.

    The default QFileIconProvider does a real shell/mime-type icon lookup
    per file, which over a network mount with thousands of entries is
    usually the dominant cost of populating the tree. This just returns one
    of two pre-built icons, built once, with no per-file OS calls.
    """

    def __init__(self) -> None:
        super().__init__()
        style = QApplication.style()
        self._folder_icon = (
            style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            if style is not None
            else QIcon()
        )
        self._file_icon = (
            style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            if style is not None
            else QIcon()
        )

    def icon(self, type_or_info) -> QIcon:  # type: ignore[override]
        if isinstance(type_or_info, QFileInfo):
            return self._folder_icon if type_or_info.isDir() else self._file_icon

        # Called with an IconType enum value (Folder/File/Computer/etc.)
        # for chrome elements like the sidebar - no per-file cost either way.
        if type_or_info == QFileIconProvider.IconType.Folder:
            return self._folder_icon
        if type_or_info == QFileIconProvider.IconType.File:
            return self._file_icon
        return super().icon(type_or_info)
