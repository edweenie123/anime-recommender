<template>
  <div class="text-center">
    <AnimeCard
      v-for="anime in animeList"
      :imgSrc="anime.imgSrc"
      :title="anime.title"
      :predScore="anime.predScore"
      :link="anime.link"
    />
  </div>
  <!-- <AnimeCard -->
  <!--   imgSrc="https://cdn.myanimelist.net/images/anime/1543/120053l.jpg" -->
  <!--   title="Hataraku Maou-sama!!" -->
  <!--   :predScore="10" -->
  <!-- /> -->
  <!-- <AnimeCard -->
  <!--   imgSrc="https://cdn.myanimelist.net/images/anime/1543/120053l.jpg" -->
  <!--   title="Hataraku Maou-sama!!" -->
  <!--   :predScore="10" -->
  <!-- /> -->
</template>

<script>
import AnimeCard from "./AnimeCard.vue";

export default {
  name: "AnimeCards",
  components: {
    AnimeCard,
  },
  props: {
    animeList: Array,
  },

  // data() {
  //   return {
  //     animeList: {}
  //   };
  // },
};
</script>
