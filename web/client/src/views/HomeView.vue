<script setup>
import Header from "../components/Header.vue";
import SearchBar from "../components/SearchBar.vue";
import AnimeCards from "../components/AnimeCards.vue";
</script>
<template>
  <Header />
  <SearchBar @upd-anime-list="updateAnimeList" />
  <hr />
  <AnimeCards :animeList="animeList" />
</template>

<script>
export default {
  data() {
    return {
      animeList: [],
    };
  },
  methods: {
    updateAnimeList(newList) {
      let titleLen = 20
      for (var item of newList) {

        // truncate anime titles
        if (item.title.length > titleLen) {
          item.title = item.title.substring(0, titleLen).trim() + "...";
        }

        // round predicted scores
        item.predScore = Math.min(item.predScore, 10);
        item.predScore = Math.round(item.predScore * 100) / 100;
      }
      this.animeList = newList;
    },
  },
};
</script>
