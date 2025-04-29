test_cases = [
    {
        "name": "Birth Year",
        "question": "When was Christopher Columbus born?",
        "context": """
        Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.
        
        The Renaissance was a period in European history that spanned from the 14th to the 17th century.
        During this time, many great artists like Leonardo da Vinci and Michelangelo created their masterpieces.
        The printing press was invented by Johannes Gutenberg around 1440, revolutionizing the spread of information.
        
        In 1492, Columbus sailed across the Atlantic Ocean, hoping to find a new route to Asia.
        His voyages were sponsored by the Catholic Monarchs of Spain, Ferdinand and Isabella.
        The Spanish Inquisition was established in 1478 to maintain Catholic orthodoxy in Spain.
        
        The Ottoman Empire was expanding during this period, capturing Constantinople in 1453.
        The Black Death had devastated Europe in the 14th century, killing millions of people.
        The Hundred Years' War between England and France ended in 1453.
        """,
        "answer": "1451"
    },
    # {
    #     "name": "Nationality",
    #     "question": "What nationality was Marie Curie?",
    #     "context": """
    #     Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.
        
    #     The field of physics saw many breakthroughs in the late 19th and early 20th centuries.
    #     Albert Einstein published his theory of relativity in 1905, revolutionizing our understanding of space and time.
    #     Niels Bohr developed the Bohr model of the atom in 1913, which explained atomic structure.
        
    #     Poland has a rich history of scientific contributions.
    #     Nicolaus Copernicus, who proposed the heliocentric model of the solar system, was also Polish.
    #     The Polish-Lithuanian Commonwealth was one of the largest and most populous countries in 16th and 17th century Europe.
        
    #     Radioactivity was first discovered by Henri Becquerel in 1896.
    #     The Curie family made significant contributions to the study of radiation.
    #     Pierre Curie, Marie's husband, was a French physicist who worked alongside her.
    #     """,
    #     "answer": "Polish"
    # },
    # {
    #     "name": "Capital City",
    #     "question": "What is the capital of Denmark?",
    #     "context": """
    #     Denmark is a Nordic country with a population of around 5.8 million. Its capital and largest city is Copenhagen.
        
    #     The Nordic countries include Sweden, Norway, Finland, Iceland, and Denmark.
    #     These countries are known for their high standard of living and social welfare systems.
    #     The Scandinavian Peninsula consists of Sweden and Norway, while Denmark is located on the Jutland Peninsula.
        
    #     Copenhagen is famous for its historic architecture and modern design.
    #     The Little Mermaid statue, based on Hans Christian Andersen's fairy tale, is a major tourist attraction.
    #     Tivoli Gardens, one of the oldest amusement parks in the world, is located in Copenhagen.
        
    #     The Danish monarchy is one of the oldest in the world, dating back to the Viking Age.
    #     Denmark is a constitutional monarchy with a parliamentary system.
    #     The country is known for its bicycle culture and environmental sustainability initiatives.
    #     """,
    #     "answer": "Copenhagen"
    # },
    # {
    #     "name": "Scientific Discovery",
    #     "question": "What did Marie Curie discover and when?",
    #     "context": """
    #     Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity.
    #     In 1898, she and her husband Pierre discovered two new elements: polonium and radium.
        
    #     The field of physics saw many breakthroughs in the late 19th and early 20th centuries.
    #     Albert Einstein published his theory of relativity in 1905, revolutionizing our understanding of space and time.
    #     Niels Bohr developed the Bohr model of the atom in 1913, which explained atomic structure.
        
    #     Radioactivity was first discovered by Henri Becquerel in 1896 when he noticed that uranium salts emitted rays that could expose photographic plates.
    #     """,
    #     "answer": "Marie Curie discovered the elements polonium and radium in 1898"
    # },
    # {
    #     "name": "Historical Event",
    #     "question": "What happened during Columbus's first voyage to the Americas?",
    #     "context": """
    #     Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.
    #     In 1492, Columbus set sail from Spain with three ships: the Santa Maria, the Pinta, and the Niña.
        
    #     The Renaissance was a period in European history that spanned from the 14th to the 17th century.
    #     During this time, many great artists like Leonardo da Vinci and Michelangelo created their masterpieces.
    #     The printing press was invented by Johannes Gutenberg around 1440, revolutionizing the spread of information.
        
    #     Columbus's first voyage lasted from August 3, 1492, to March 15, 1493.
    #     He landed in the Bahamas on October 12, 1492, thinking he had reached Asia.
    #     The Santa Maria ran aground on Christmas Day 1492 and had to be abandoned.
        
    #     The Ottoman Empire was expanding during this period, capturing Constantinople in 1453.
    #     The Black Death had devastated Europe in the 14th century, killing millions of people.
    #     The Hundred Years' War between England and France ended in 1453.
    #     """,
    #     "answer": "Columbus sailed from Spain with three ships, landed in the Bahamas on October 12, 1492, and the Santa Maria was lost on Christmas Day"
    # },
    # {
    #     "name": "Geographical Feature",
    #     "question": "What are some notable geographical features of Denmark?",
    #     "context": """
    #     Denmark is a Nordic country with a population of around 5.8 million. Its capital and largest city is Copenhagen.
        
    #     The Nordic countries include Sweden, Norway, Finland, Iceland, and Denmark.
    #     These countries are known for their high standard of living and social welfare systems.
    #     The Scandinavian Peninsula consists of Sweden and Norway, while Denmark is located on the Jutland Peninsula.
        
    #     Denmark consists of the Jutland Peninsula and over 400 islands, with Zealand being the largest.
    #     The country is mostly flat, with its highest point being Møllehøj at 170.86 meters above sea level.
    #     Denmark has a long coastline and is surrounded by the North Sea and the Baltic Sea.
        
    #     The Danish monarchy is one of the oldest in the world, dating back to the Viking Age.
    #     Denmark is a constitutional monarchy with a parliamentary system.
    #     The country is known for its bicycle culture and environmental sustainability initiatives.
    #     """,
    #     "answer": "Denmark consists of the Jutland Peninsula and over 400 islands, with Zealand being the largest, and has a mostly flat landscape with its highest point at Møllehøj"
    # },
    # {
    #     "name": "Moon Landing",
    #     "question": "Who was the first person to walk on the moon?",
    #     "context": """
    #     The space race between the United States and Soviet Union was a major part of the Cold War.
    #     Yuri Gagarin became the first human in space in 1961, a major victory for the Soviet Union.
    #     The Apollo program was NASA's response, aiming to land humans on the moon.
        
    #     Many astronauts trained for the moon landing, including Buzz Aldrin, Michael Collins, and Neil Armstrong.
    #     The Soviet Union's Luna program successfully landed robotic spacecraft on the moon in 1966.
    #     Some conspiracy theorists believe the moon landing was faked in a Hollywood studio.
        
    #     The Apollo 11 mission launched on July 16, 1969, with three astronauts aboard.
    #     The lunar module Eagle landed on the moon's surface on July 20, 1969.
    #     Buzz Aldrin was the second person to step onto the lunar surface.
    #     """,
    #     "answer": "Neil Armstrong was first"
    # },
    # {
    #     "name": "Invention Year",
    #     "question": "When was the telephone invented?",
    #     "context": """
    #     The 19th century saw many important technological innovations that changed communication forever.
    #     Samuel Morse invented the telegraph in 1837, allowing messages to be sent over long distances.
    #     Thomas Edison developed the phonograph in 1877, which could record and play back sound.
        
    #     Many inventors were working on voice transmission devices in the late 1800s.
    #     Elisha Gray filed a patent for a telephone-like device on the same day as Alexander Graham Bell.
    #     Antonio Meucci claimed to have invented the telephone in 1854, but couldn't afford the patent.
        
    #     The first successful telephone call was made in 1876.
    #     Bell's patent was granted on March 7, 1876, just hours before Gray's application.
    #     The first words spoken over the telephone were "Mr. Watson, come here, I want to see you."
    #     """,
    #     "answer": "The telephone was invented in 1876"
    # },
    # {
    #     "name": "Famous Painting",
    #     "question": "Who painted the Mona Lisa?",
    #     "context": """
    #     The Renaissance period produced many famous artists and masterpieces.
    #     Michelangelo painted the Sistine Chapel ceiling between 1508 and 1512.
    #     Raphael created the School of Athens fresco in 1511.
        
    #     Leonardo da Vinci was a polymath who worked in many fields.
    #     He designed flying machines and studied human anatomy in detail.
    #     His notebooks contain thousands of pages of drawings and observations.
        
    #     The Mona Lisa is one of the most famous paintings in the world.
    #     It was stolen from the Louvre in 1911 and recovered in 1913.
    #     The painting's enigmatic smile has fascinated viewers for centuries.
    #     """,
    #     "answer": "Leonardo da Vinci painted it"
    # },
    # {
    #     "name": "Planet Discovery",
    #     "question": "When was Pluto discovered?",
    #     "context": """
    #     The outer solar system has been a source of many astronomical discoveries.
    #     Neptune was discovered in 1846 after mathematical predictions of its existence.
    #     Uranus was found in 1781 by William Herschel using a telescope.
        
    #     Many astronomers searched for a ninth planet in the early 20th century.
    #     Percival Lowell predicted the existence of Planet X based on orbital irregularities.
    #     Clyde Tombaugh used Lowell's calculations to guide his search for the new planet.
        
    #     Pluto was reclassified as a dwarf planet in 2006.
    #     The discovery of Eris in 2005 led to the redefinition of what constitutes a planet.
    #     Pluto's moon Charon was discovered in 1978.
    #     """,
    #     "answer": "Pluto was discovered in 1930"
    # }
]