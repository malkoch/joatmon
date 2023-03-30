from joatmon.search.wikipedia import (
    languages,
    set_lang,
    summary
)

print(languages())
set_lang('tr')
print(summary("Facebook", sentences=1))
