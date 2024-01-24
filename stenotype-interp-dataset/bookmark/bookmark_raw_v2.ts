interface BookmarkBefore {
    name: string
    url: string
    tags: string
    views: string
    discription: string
}

export const mainView = 'Home'

const bookmarksBefore: BookmarkBefore[] = [
    {
        name: 'Google',
        url: 'https://www.google.com',
        tags: 'Search',
        views: 'Home',
        discription: '',
    },
    {
        name: 'React',
        url: 'https://zh-hans.reactjs.org',
        tags: 'Web',
        views: 'Develop',
        discription: '',
    },
    {
        name: 'Python Docs',
        url: 'https://docs.python.org/3/',
        tags: 'Python',
        views: '',
        discription: '',
    },
    {
        name: 'PyPI',
        url: 'https://pypi.org',
        tags: 'Python',
        views: '',
        discription: '',
    },
    {
        name: 'VSCodeThemes',
        url: 'https://vscodethemes.com',
        tags: 'Tools',
        views: '',
        discription: 'VSCode主题',
    },
]

const regexp = new RegExp('https{0,1}://.*?/')

function getFavicon(url: string) {
    let favicon = url.match(regexp)
    if (favicon) return favicon[0] + 'favicon.ico'
    else return url + '/favicon.ico'
}

export interface Bookmark {
    name: string
    url: string
    img: string
    tags: string | null
    views: string | null
    discription: string
}

export const bookmarks: Bookmark[] = bookmarksBefore.map(item => {
    return {
        name: item.name,
        url: item.url,
        img: getFavicon(item.url),
        tags: item.tags,
        views: item.views,
        discription: item.discription,
    }
})