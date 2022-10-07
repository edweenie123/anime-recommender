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
      this.animeList = newList;
      console.log(this.animeList);
    },
  },
};
</script>
