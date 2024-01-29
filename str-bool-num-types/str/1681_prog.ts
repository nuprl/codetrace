export type HeaderModelName = 'default' | 'headphones' | 'drawing' | 'dude'

export interface SingleProjectModel {
  icon: '🤖'
  companyLogo: <FILL>
  name: string
  url: string
  description: string
  status: 'done' | 'in-progress' | 'to-do'
}

export interface RepositoryModel {
  title: String
  subtitle: String
  seeMore: String
  url: String
  collection: SingleProjectModel[]
}

export const RepositoriesData = (): RepositoryModel[] => [
  {
    title: 'Featured work',
    subtitle: 'Some projects that I’ve finally finished... almost',
    seeMore: 'I might have more',
    url: 'https://www.linkedin.com/in/wikz87/',
    collection: [
      {
        icon: '🤖',
        companyLogo: 'logo-achieve',
        name: 'Achieve Test Prep',
        url: 'https://prep.achievetestprep.com/',
        description:
          'A learning management system to help the students to achieve more in less time and using less money.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: '1on1 Quiz',
        url: 'https://1on1quiz.com',
        description: '1on1 Quiz - Competitive gaming platform',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Content Management System',
        url: 'https://rcms.beaconhouse.net/',
        description: 'A multi platforms(LMS) supported content management system',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-git',
        name: 'Portfolio Website',
        url: 'https://github.com/wikz/Portfolio-Website',
        description: "This one. Yep, you can check the cool stuff that I've used to build it",
        status: 'in-progress'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-git',
        name: 'Resource Preloader',
        url: 'https://github.com/wikz/Resource-Preloader',
        description:
          'A JavaScript based powerful resource pre-loader with concurrent download and dependency resolving feature.',
        status: 'done'
      }
    ]
  },
  {
    title: 'Mobile Applications',
    subtitle: 'Some apps that I built',
    seeMore: 'I might have more',
    url: 'https://www.linkedin.com/in/wikz87/',
    collection: [
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Learn Smart Pakistan',
        url: 'https://play.google.com/store/apps/details?id=com.knowledgeplatform.lsp',
        description:
          'A cross-platform mobile application for Ultrabot(Learning Management System) with data storing and synching capability. Currently serving 250k+ students.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Ultrabot Prime',
        url: 'https://play.google.com/store/apps/details?id=com.knowledgeplatform.ultrabotprime',
        description: 'A cross-platform mobile LMS App created for M1 Singapore.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Literacy & Numeracy Placement',
        url: '',
        description: 'A multilingual app built in collaboration with World bank and used by 50k+ individual users.',
        status: 'done'
      }
    ]
  },
  {
    title: 'Applications',
    subtitle: "Some Web & desktop Applications I've worked on",
    seeMore: 'I might have more',
    url: 'https://www.linkedin.com/in/wikz87/',
    collection: [
      {
        icon: '🤖',
        companyLogo: 'logo-achieve',
        name: 'Achieve Test Prep',
        url: 'https://prep.achievetestprep.com/',
        description:
          'A learning management system to help the students to achieve more in less time and using less money.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: '1on1 Quiz',
        url: 'https://1on1quiz.com',
        description: '1on1 Quiz - Competitive gaming platform',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Content Management System',
        url: 'https://rcms.beaconhouse.net/',
        description: 'A multi platforms(LMS) supported content management system',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-ewi',
        name: 'Voice Activated FM Audio Broadcast Logger',
        url: '#',
        description: 'An Audio logging software for Radio transmissions. Logs audio transmission in mp3 audio clips.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-ewi',
        name: 'Audio Broadcast Profanity Delay System',
        url: '#',
        description: 'A software being used in Radio transmission broadcasting industry.',
        status: 'done'
      }
    ]
  },
  {
    title: 'CrossPlatform Games',
    subtitle: 'Some games that I’ve launched successfully',
    seeMore: 'I might have more',
    url: 'https://www.linkedin.com/in/wikz87/',
    collection: [
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Dino Eggs',
        url: 'http://gamesdemo.knowledgeplatform.com/demos/dino-eggs/',
        description:
          'A multi-level online game covering 120+ mastery topics of mathematics for middle school students.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Factor Monsters',
        url: 'https://apps.apple.com/pk/app/factor-monsters/id987903121',
        description: 'A drill to master different skills pertaining to factorization.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Mind Tussle',
        url: 'https://apps.apple.com/pk/app/mind-tussle/id1052659847',
        description: 'Improve mental math in an online 1-on-1, time-bound 3 levels game.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Martian Multiples',
        url: 'https://apps.apple.com/pk/app/martian-multiples/id1101691574',
        description: 'A game to master skills of finding Least common multiples and Highest common factors.',
        status: 'done'
      },
      {
        icon: '🤖',
        companyLogo: 'logo-kp',
        name: 'Kampong',
        url: 'http://gamesdemo.knowledgeplatform.com/demos/kampong/',
        description: 'A card game based on adaptive templates for primary content.',
        status: 'done'
      }
    ]
  }
]
